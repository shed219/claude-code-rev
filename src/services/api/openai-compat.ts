/**
 * OpenAI API compatibility layer for claude-code.
 *
 * This module translates between the Anthropic API format (used throughout
 * the codebase) and generic OpenAI-compatible API format, enabling claude-code
 * to work with any OpenAI-compatible endpoint.
 *
 * Activated when OPENAI_API_KEY is set, or when CLAUDE_CODE_USE_OPENAI=true
 * is set (e.g. for endpoints that use a different auth mechanism).
 *
 * Configuration environment variables:
 *   OPENAI_API_KEY        - API key for the OpenAI-compatible endpoint
 *   OPENAI_BASE_URL       - Base URL (default: https://api.openai.com/v1)
 *   CLAUDE_CODE_USE_OPENAI - Set to 'true' to force OpenAI mode (requires OPENAI_BASE_URL)
 */

import type {
  BetaContentBlock,
  BetaContentBlockParam,
  BetaImageBlockParam,
  BetaMessage,
  BetaMessageDeltaUsage,
  BetaMessageStreamParams,
  BetaRawMessageStreamEvent,
  BetaRequestDocumentBlock,
  BetaToolResultBlockParam,
  BetaToolUnion,
  BetaUsage,
  BetaMessageParam as MessageParam,
} from '@anthropic-ai/sdk/resources/beta/messages/messages.mjs'
import type { TextBlockParam } from '@anthropic-ai/sdk/resources/index.mjs'
import { randomUUID } from 'crypto'
import { logForDebugging } from '../../utils/debug.js'

// ─── OpenAI wire types ────────────────────────────────────────────────────────

interface OpenAITextContentPart {
  type: 'text'
  text: string
}

interface OpenAIImageContentPart {
  type: 'image_url'
  image_url: { url: string; detail?: string }
}

type OpenAIContentPart = OpenAITextContentPart | OpenAIImageContentPart

interface OpenAIToolCall {
  id: string
  type: 'function'
  function: { name: string; arguments: string }
}

interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool'
  content: string | OpenAIContentPart[] | null
  tool_calls?: OpenAIToolCall[]
  tool_call_id?: string
}

interface OpenAITool {
  type: 'function'
  function: {
    name: string
    description?: string
    parameters?: Record<string, unknown>
  }
}

interface OpenAIStreamChunk {
  id: string
  object: string
  choices: {
    index: number
    delta: {
      role?: string
      content?: string | null
      tool_calls?: {
        index: number
        id?: string
        type?: string
        function?: { name?: string; arguments?: string }
      }[]
    }
    finish_reason: string | null
  }[]
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  } | null
}

interface OpenAIResponse {
  id: string
  object: string
  choices: {
    index: number
    message: {
      role: string
      content: string | null
      tool_calls?: OpenAIToolCall[]
    }
    finish_reason: string
  }[]
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
  }
}

// ─── Request translation (Anthropic → OpenAI) ────────────────────────────────

function convertContentBlockToOpenAIPart(
  block: BetaContentBlockParam,
): OpenAIContentPart | null {
  switch (block.type) {
    case 'text':
      return { type: 'text', text: block.text }
    case 'image': {
      const src = (block as BetaImageBlockParam).source
      if (src.type === 'base64') {
        return {
          type: 'image_url',
          image_url: { url: `data:${src.media_type};base64,${src.data}` },
        }
      }
      if (src.type === 'url') {
        return { type: 'image_url', image_url: { url: src.url } }
      }
      return null
    }
    case 'document': {
      // Represent document as text where possible
      const doc = block as BetaRequestDocumentBlock
      const textSource = doc.source as { text?: string }
      if (textSource?.text) {
        return { type: 'text', text: textSource.text }
      }
      return null
    }
    default:
      return null
  }
}

/**
 * Convert Anthropic-format messages + system prompt to OpenAI messages array.
 * Tool result blocks (which live inside user messages in the Anthropic format)
 * are extracted and emitted as separate 'tool' role messages, matching OpenAI's
 * convention.
 */
export function convertMessagesToOpenAI(
  messages: MessageParam[],
  system?: string | TextBlockParam[],
): OpenAIMessage[] {
  const result: OpenAIMessage[] = []

  if (system) {
    let systemText: string
    if (typeof system === 'string') {
      systemText = system
    } else {
      systemText = (system as TextBlockParam[])
        .filter(b => b.type === 'text')
        .map(b => b.text)
        .join('\n')
    }
    if (systemText.trim()) {
      result.push({ role: 'system', content: systemText })
    }
  }

  for (const msg of messages) {
    if (typeof msg.content === 'string') {
      result.push({ role: msg.role as 'user' | 'assistant', content: msg.content })
      continue
    }

    if (!Array.isArray(msg.content)) {
      result.push({ role: msg.role as 'user' | 'assistant', content: '' })
      continue
    }

    if (msg.role === 'user') {
      const toolResults: BetaToolResultBlockParam[] = []
      const parts: OpenAIContentPart[] = []

      for (const block of msg.content) {
        if (block.type === 'tool_result') {
          toolResults.push(block as BetaToolResultBlockParam)
        } else {
          const part = convertContentBlockToOpenAIPart(block)
          if (part) parts.push(part)
        }
      }

      // Emit tool results as individual 'tool' messages
      for (const tr of toolResults) {
        let content: string
        if (typeof tr.content === 'string') {
          content = tr.content
        } else if (Array.isArray(tr.content)) {
          content = tr.content
            .filter(c => c.type === 'text')
            .map(c => (c as { text: string }).text)
            .join('\n')
        } else {
          content = ''
        }
        result.push({ role: 'tool', tool_call_id: tr.tool_use_id, content })
      }

      // Emit user message if there is non-tool-result content
      if (parts.length > 0) {
        // Use plain string when the only content is a single text block
        const firstPart = parts[0]
        if (parts.length === 1 && firstPart?.type === 'text') {
          result.push({ role: 'user', content: (firstPart as OpenAITextContentPart).text })
        } else {
          result.push({ role: 'user', content: parts })
        }
      }
    } else if (msg.role === 'assistant') {
      let textContent = ''
      const toolCalls: OpenAIToolCall[] = []

      for (const block of msg.content) {
        if (block.type === 'text') {
          textContent += (block as { text: string }).text
        } else if (block.type === 'tool_use') {
          const tu = block as {
            id: string
            name: string
            input: Record<string, unknown>
          }
          toolCalls.push({
            id: tu.id,
            type: 'function',
            function: {
              name: tu.name,
              arguments: JSON.stringify(tu.input),
            },
          })
        }
        // thinking, server_tool_use, and other Anthropic-specific blocks are dropped
      }

      const openAIMsg: OpenAIMessage = {
        role: 'assistant',
        content: textContent || null,
      }
      if (toolCalls.length > 0) {
        openAIMsg.tool_calls = toolCalls
      }
      result.push(openAIMsg)
    }
  }

  return result
}

/**
 * Convert Anthropic tool definitions to OpenAI function tool format.
 * Built-in Anthropic-server tools (computer use, bash, etc.) are filtered out
 * since OpenAI endpoints do not recognise them.
 */
function convertToolsToOpenAI(tools: BetaToolUnion[]): OpenAITool[] {
  return tools
    .filter(tool => {
      // Keep only user-defined tools that have input_schema (custom tools)
      const t = tool as Record<string, unknown>
      return typeof t['name'] === 'string' && 'input_schema' in t
    })
    .map(tool => {
      const t = tool as {
        name: string
        description?: string
        input_schema?: Record<string, unknown>
      }
      return {
        type: 'function' as const,
        function: {
          name: t.name,
          ...(t.description && { description: t.description }),
          ...(t.input_schema && { parameters: t.input_schema }),
        },
      }
    })
}

function convertToolChoice(
  toolChoice: BetaMessageStreamParams['tool_choice'],
): string | { type: 'function'; function: { name: string } } | undefined {
  if (!toolChoice) return undefined
  switch (toolChoice.type) {
    case 'auto':
      return 'auto'
    case 'any':
      return 'required'
    case 'none':
      return 'none'
    case 'tool':
      return { type: 'function', function: { name: (toolChoice as { name: string }).name } }
    default:
      return 'auto'
  }
}

// ─── Response translation (OpenAI → Anthropic) ───────────────────────────────

function convertStopReason(finishReason: string | null): BetaMessage['stop_reason'] {
  switch (finishReason) {
    case 'tool_calls':
      return 'tool_use'
    case 'length':
      return 'max_tokens'
    case 'stop':
    default:
      return 'end_turn'
  }
}

function convertOpenAIResponseToAnthropicMessage(
  response: OpenAIResponse,
  model: string,
): BetaMessage {
  const choice = response.choices[0]!
  const msg = choice.message
  const content: BetaContentBlock[] = []

  if (msg.content) {
    content.push({ type: 'text', text: msg.content })
  }
  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      let input: Record<string, unknown> = {}
      try {
        input = JSON.parse(tc.function.arguments)
      } catch {
        // leave as empty object
      }
      content.push({
        type: 'tool_use',
        id: tc.id,
        name: tc.function.name,
        input,
      })
    }
  }

  const usage = response.usage ?? { prompt_tokens: 0, completion_tokens: 0 }
  return {
    id: response.id || `msg_${randomUUID()}`,
    type: 'message',
    role: 'assistant',
    content,
    model,
    stop_reason: convertStopReason(choice.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens ?? 0,
      output_tokens: usage.completion_tokens ?? 0,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    } as BetaUsage,
  }
}

// ─── Streaming translation ────────────────────────────────────────────────────

/**
 * Read an OpenAI SSE stream and yield Anthropic-format streaming events.
 *
 * Anthropic event sequence for a text response:
 *   message_start → content_block_start(text,0) → content_block_delta* →
 *   content_block_stop(0) → message_delta → message_stop
 *
 * For tool calls the text block (if any) is closed before tool_use blocks
 * are opened, matching the order the SDK consumer expects.
 */
async function* translateOpenAIStreamToAnthropicEvents(
  response: Response,
  model: string,
  messageId: string,
): AsyncGenerator<BetaRawMessageStreamEvent> {
  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  let messageEmitted = false
  let textBlockIndex: number | null = null
  // Map: OpenAI tool_call.index → Anthropic content_block index
  const toolBlockMap = new Map<number, { anthropicIdx: number; closed: boolean }>()
  let nextBlockIndex = 0
  let inputTokens = 0
  let outputTokens = 0
  // Track which content blocks have been closed with content_block_stop
  const closedBlocks = new Set<number>()

  const emitMessageStart = (chunkId: string): BetaRawMessageStreamEvent => ({
    type: 'message_start',
    message: {
      id: chunkId || messageId,
      type: 'message',
      role: 'assistant',
      content: [],
      model,
      stop_reason: null,
      stop_sequence: null,
      usage: {
        input_tokens: 0,
        output_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
      } as BetaUsage,
    },
  })

  const closeTextBlock = function* (): Generator<BetaRawMessageStreamEvent> {
    if (textBlockIndex !== null && !closedBlocks.has(textBlockIndex)) {
      yield { type: 'content_block_stop', index: textBlockIndex }
      closedBlocks.add(textBlockIndex)
      textBlockIndex = null
    }
  }

  const closeAllBlocks = function* (): Generator<BetaRawMessageStreamEvent> {
    yield* closeTextBlock()
    for (const [, entry] of toolBlockMap) {
      if (!closedBlocks.has(entry.anthropicIdx)) {
        yield { type: 'content_block_stop', index: entry.anthropicIdx }
        closedBlocks.add(entry.anthropicIdx)
      }
    }
  }

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        const trimmed = line.trim()
        if (!trimmed.startsWith('data: ')) continue
        const data = trimmed.slice(6).trim()

        if (data === '[DONE]') {
          if (!messageEmitted) {
            messageEmitted = true
            yield emitMessageStart(messageId)
          }
          yield* closeAllBlocks()
          yield {
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: outputTokens } as BetaMessageDeltaUsage,
          }
          yield { type: 'message_stop' }
          return
        }

        let chunk: OpenAIStreamChunk
        try {
          chunk = JSON.parse(data)
        } catch {
          continue
        }

        // Emit message_start once
        if (!messageEmitted) {
          messageEmitted = true
          yield emitMessageStart(chunk.id)
        }

        // Accumulate usage
        if (chunk.usage) {
          inputTokens = chunk.usage.prompt_tokens ?? inputTokens
          outputTokens = chunk.usage.completion_tokens ?? outputTokens
        }

        const choice = chunk.choices?.[0]
        if (!choice) continue
        const delta = choice.delta

        // Handle text delta
        if (delta?.content) {
          if (textBlockIndex === null) {
            textBlockIndex = nextBlockIndex++
            yield {
              type: 'content_block_start',
              index: textBlockIndex,
              content_block: { type: 'text', text: '' },
            }
          }
          yield {
            type: 'content_block_delta',
            index: textBlockIndex,
            delta: { type: 'text_delta', text: delta.content },
          }
        }

        // Handle tool call deltas
        if (delta?.tool_calls?.length) {
          // Close any open text block first
          yield* closeTextBlock()

          for (const toolDelta of delta.tool_calls) {
            const oaiIdx = toolDelta.index ?? 0

            if (!toolBlockMap.has(oaiIdx)) {
              // First chunk for this tool call — emit content_block_start
              const anthropicIdx = nextBlockIndex++
              toolBlockMap.set(oaiIdx, { anthropicIdx, closed: false })
              yield {
                type: 'content_block_start',
                index: anthropicIdx,
                content_block: {
                  type: 'tool_use',
                  id: toolDelta.id || `toolu_${randomUUID()}`,
                  name: toolDelta.function?.name || '',
                  input: {} as Record<string, unknown>,
                },
              }
            }

            const entry = toolBlockMap.get(oaiIdx)!
            if (toolDelta.function?.arguments) {
              yield {
                type: 'content_block_delta',
                index: entry.anthropicIdx,
                delta: {
                  type: 'input_json_delta',
                  partial_json: toolDelta.function.arguments,
                },
              }
            }
          }
        }

        // Handle finish reason
        if (choice.finish_reason) {
          yield* closeAllBlocks()
          yield {
            type: 'message_delta',
            delta: {
              stop_reason: convertStopReason(choice.finish_reason),
              stop_sequence: null,
            },
            usage: { output_tokens: outputTokens } as BetaMessageDeltaUsage,
          }
          yield { type: 'message_stop' }
          return
        }
      }
    }
  } finally {
    try {
      reader.releaseLock()
    } catch {
      // already released
    }
  }

  // Reached end of stream without an explicit finish_reason
  if (messageEmitted) {
    yield* closeAllBlocks()
    yield {
      type: 'message_delta',
      delta: { stop_reason: 'end_turn', stop_sequence: null },
      usage: { output_tokens: outputTokens } as BetaMessageDeltaUsage,
    }
    yield { type: 'message_stop' }
  }
}

// ─── HTTP helpers ─────────────────────────────────────────────────────────────

function getOpenAIBaseUrl(): string {
  return (process.env.OPENAI_BASE_URL ?? 'https://api.openai.com/v1').replace(/\/$/, '')
}

function getOpenAIApiKey(): string {
  return process.env.OPENAI_API_KEY ?? ''
}

async function fetchOpenAI(
  body: Record<string, unknown>,
  options: { signal?: AbortSignal; timeout?: number; headers?: Record<string, string> },
): Promise<Response> {
  const baseUrl = getOpenAIBaseUrl()
  const apiKey = getOpenAIApiKey()

  const requestHeaders: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
    ...options.headers,
  }

  logForDebugging(`[OpenAI compat] POST ${baseUrl}/chat/completions model=${body['model']}`)

  const fetchOptions: RequestInit = {
    method: 'POST',
    headers: requestHeaders,
    body: JSON.stringify(body),
    signal: options.signal,
  }

  const response = await fetch(`${baseUrl}/chat/completions`, fetchOptions)

  if (!response.ok && !body['stream']) {
    const errorText = await response.text()
    throw new Error(
      `OpenAI API error ${response.status}: ${errorText}`,
    )
  }

  return response
}

// ─── Build OpenAI request body ────────────────────────────────────────────────

function buildOpenAIRequestBody(
  params: BetaMessageStreamParams & { stream?: boolean },
): Record<string, unknown> {
  const messages = convertMessagesToOpenAI(
    params.messages,
    params.system as string | TextBlockParam[] | undefined,
  )

  const tools = params.tools?.length
    ? convertToolsToOpenAI(params.tools)
    : undefined

  const toolChoice = convertToolChoice(params.tool_choice)

  const body: Record<string, unknown> = {
    model: params.model,
    messages,
    max_tokens: params.max_tokens,
    ...(params.temperature !== undefined && { temperature: params.temperature }),
    ...(params.top_p !== undefined && { top_p: params.top_p }),
    ...(tools?.length && { tools }),
    ...(toolChoice !== undefined && { tool_choice: toolChoice }),
    ...(params.stream && {
      stream: true,
      // Request usage info in the final streaming chunk
      stream_options: { include_usage: true },
    }),
  }

  return body
}

// ─── Fake Anthropic client ────────────────────────────────────────────────────

/**
 * Returns a duck-typed object that satisfies the subset of the Anthropic SDK
 * interface used by this codebase, but routes all requests through an
 * OpenAI-compatible API endpoint.
 *
 * The returned object is cast to `unknown` so callers can cast it to
 * `Anthropic` without TypeScript complaining.
 */
export function createOpenAICompatClient(): unknown {
  const betaMessages = {
    create(
      params: BetaMessageStreamParams & { stream?: boolean },
      options: {
        signal?: AbortSignal
        timeout?: number
        headers?: Record<string, string>
      } = {},
    ) {
      if (params.stream) {
        // Streaming path: return an object with .withResponse()
        return {
          withResponse: async () => {
            const body = buildOpenAIRequestBody(params)
            const response = await fetchOpenAI(body, options)

            if (!response.ok) {
              const errorText = await response.text()
              throw new Error(`OpenAI API error ${response.status}: ${errorText}`)
            }

            const messageId = `msg_${randomUUID()}`

            // Create an async iterable from the SSE stream
            const data = {
              [Symbol.asyncIterator]: async function* () {
                yield* translateOpenAIStreamToAnthropicEvents(
                  response,
                  params.model,
                  messageId,
                )
              },
            }

            return {
              data,
              response,
              request_id:
                response.headers.get('x-request-id') ||
                response.headers.get('request-id') ||
                messageId,
            }
          },
        }
      }

      // Non-streaming path: return a Promise<BetaMessage>
      return (async () => {
        const body = buildOpenAIRequestBody(params)
        const response = await fetchOpenAI(body, options)
        const json: OpenAIResponse = await response.json()
        return convertOpenAIResponseToAnthropicMessage(json, params.model)
      })()
    },
  }

  return {
    beta: {
      messages: betaMessages,
    },
  }
}
