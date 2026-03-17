/**
 * iFlowCLIAgent: iFlow CLI SDK-based observation extraction
 *
 * Alternative to SDKAgent that uses the iFlow CLI SDK directly
 * for extracting observations from tool usage.
 *
 * Responsibility:
 * - Call iFlow CLI SDK for observation extraction
 * - Parse XML responses (same format as Claude)
 * - Sync to database and Chroma
 */

import path from 'path';
import { homedir } from 'os';
import { DatabaseManager } from './DatabaseManager.js';
import { SessionManager } from './SessionManager.js';
import { logger } from '../../utils/logger.js';
import { buildInitPrompt, buildObservationPrompt, buildSummaryPrompt, buildContinuationPrompt } from '../../sdk/prompts.js';
import { SettingsDefaultsManager } from '../../shared/SettingsDefaultsManager.js';
import type { ActiveSession, ConversationMessage } from '../worker-types.js';
import { ModeManager } from '../domain/ModeManager.js';
import {
  processAgentResponse,
  shouldFallbackToClaude,
  isAbortError,
  type WorkerRef,
  type FallbackAgent
} from './agents/index.js';

// iFlow CLI SDK imports
// @ts-ignore - SDK types may not be available during development
import { IFlowClient, MessageType, IFlowOptions, MCPServerConfig } from '@iflow-ai/iflow-cli-sdk';

// Context window management constants
const DEFAULT_MAX_CONTEXT_MESSAGES = 30;
const DEFAULT_MAX_ESTIMATED_TOKENS = 100000;
const CHARS_PER_TOKEN_ESTIMATE = 4;

export class iFlowCLIAgent {
  private dbManager: DatabaseManager;
  private sessionManager: SessionManager;
  private fallbackAgent: FallbackAgent | null = null;
  private client: IFlowClient | null = null;

  constructor(dbManager: DatabaseManager, sessionManager: SessionManager) {
    this.dbManager = dbManager;
    this.sessionManager = sessionManager;
  }

  /**
   * Set the fallback agent (Claude SDK) for when iFlow CLI fails
   */
  setFallbackAgent(agent: FallbackAgent): void {
    this.fallbackAgent = agent;
  }

  /**
   */

  /**
   * Start iFlow CLI agent for a session
   */
  async startSession(session: ActiveSession, worker?: WorkerRef): Promise<void> {
    try {
      // Check authentication
      if (!isiFlowCLIAvailable()) {
        throw new Error('iFlow CLI not authenticated. Please run "iflow login" to authenticate.');
      }

      // Generate synthetic memorySessionId (iFlow CLI is stateless at the observation level)
      if (!session.memorySessionId) {
        const syntheticMemorySessionId = `iflow-${session.contentSessionId}-${Date.now()}`;
        session.memorySessionId = syntheticMemorySessionId;
        this.dbManager.getSessionStore().updateMemorySessionId(session.sessionDbId, syntheticMemorySessionId);
        logger.info('SESSION', `MEMORY_ID_GENERATED | sessionDbId=${session.sessionDbId} | provider=iFlowCLI`);
      }

      // Initialize iFlow client with default options
      // SDK will automatically use credentials from ~/.iflow/settings.json
      const mcpServerConfig: MCPServerConfig = {
        name: 'claude-mem',
        command: 'echo',
        args: ['noop'],
        env: [],
      };

      const options: IFlowOptions = {
        transportMode: 'stdio',
        logLevel: 'INFO',
        timeout: 300000, // 5 minutes
        mcpServers: [mcpServerConfig],
        sessionSettings: {
          system_prompt: undefined,
          permission_mode: 'yolo',
        },
      };

      this.client = new IFlowClient(options);
      await this.client.connect();

      // Load active mode
      const mode = ModeManager.getInstance().getActiveMode();

      // Build initial prompt
      const initPrompt = session.lastPromptNumber === 1
        ? buildInitPrompt(session.project, session.contentSessionId, session.userPrompt, mode)
        : buildContinuationPrompt(session.userPrompt, session.lastPromptNumber, session.contentSessionId, mode);

      // Add to conversation history and query iFlow with full context
      session.conversationHistory.push({ role: 'user', content: initPrompt });
      const initResponse = await this.queryiFlowMultiTurn(session.conversationHistory);

      if (initResponse.content) {
        // Track token usage
        const tokensUsed = initResponse.tokensUsed || 0;
        session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
        session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);

        // Process response using shared ResponseProcessor
        await processAgentResponse(
          initResponse.content,
          session,
          this.dbManager,
          this.sessionManager,
          worker,
          tokensUsed,
          null,
          'iFlowCLI',
          undefined
        );
      } else {
        logger.error('SDK', 'Empty iFlow CLI init response - session may lack context', {
          sessionId: session.sessionDbId,
        });
      }

      // Track cwd from messages for CLAUDE.md generation
      let lastCwd: string | undefined;

      // Process pending messages
      for await (const message of this.sessionManager.getMessageIterator(session.sessionDbId)) {
        session.processingMessageIds.push(message._persistentId);

        // Capture cwd from each message for worktree support
        if (message.cwd) {
          lastCwd = message.cwd;
        }

        // Capture earliest timestamp BEFORE processing
        const originalTimestamp = session.earliestPendingTimestamp;

        if (message.type === 'observation') {
          // Update last prompt number
          if (message.prompt_number !== undefined) {
            session.lastPromptNumber = message.prompt_number;
          }

          // CRITICAL: Check memorySessionId BEFORE making expensive LLM call
          if (!session.memorySessionId) {
            throw new Error('Cannot process observations: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          // Build observation prompt
          const obsPrompt = buildObservationPrompt({
            id: 0,
            tool_name: message.tool_name!,
            tool_input: JSON.stringify(message.tool_input),
            tool_output: JSON.stringify(message.tool_response),
            created_at_epoch: originalTimestamp ?? Date.now(),
            cwd: message.cwd
          });

          // Add to conversation history and query iFlow with full context
          session.conversationHistory.push({ role: 'user', content: obsPrompt });
          const obsResponse = await this.queryiFlowMultiTurn(session.conversationHistory);

          let tokensUsed = 0;
          if (obsResponse.content) {
            tokensUsed = obsResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            obsResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'iFlowCLI',
            lastCwd
          );

        } else if (message.type === 'summarize') {
          // CRITICAL: Check memorySessionId BEFORE making expensive LLM call
          if (!session.memorySessionId) {
            throw new Error('Cannot process summary: memorySessionId not yet captured. This session may need to be reinitialized.');
          }

          // Build summary prompt
          const summaryPrompt = buildSummaryPrompt({
            id: session.sessionDbId,
            memory_session_id: session.memorySessionId,
            project: session.project,
            user_prompt: session.userPrompt,
            last_assistant_message: message.last_assistant_message || ''
          }, mode);

          // Add to conversation history and query iFlow with full context
          session.conversationHistory.push({ role: 'user', content: summaryPrompt });
          const summaryResponse = await this.queryiFlowMultiTurn(session.conversationHistory);

          let tokensUsed = 0;
          if (summaryResponse.content) {
            tokensUsed = summaryResponse.tokensUsed || 0;
            session.cumulativeInputTokens += Math.floor(tokensUsed * 0.7);
            session.cumulativeOutputTokens += Math.floor(tokensUsed * 0.3);
          }

          // Process response using shared ResponseProcessor
          await processAgentResponse(
            summaryResponse.content || '',
            session,
            this.dbManager,
            this.sessionManager,
            worker,
            tokensUsed,
            originalTimestamp,
            'iFlowCLI',
            lastCwd
          );
        }
      }

      // Mark session complete
      const sessionDuration = Date.now() - session.startTime;
      logger.success('SDK', 'iFlow CLI agent completed', {
        sessionId: session.sessionDbId,
        duration: `${(sessionDuration / 1000).toFixed(1)}s`,
        historyLength: session.conversationHistory.length,
      });

    } catch (error: unknown) {
      // Cleanup client on error
      if (this.client) {
        try {
          await this.client.disconnect();
        } catch {
          // Ignore disconnect errors
        }
        this.client = null;
      }

      if (isAbortError(error)) {
        logger.warn('SDK', 'iFlow CLI agent aborted', { sessionId: session.sessionDbId });
        throw error;
      }

      // Check if we should fall back to Claude
      if (shouldFallbackToClaude(error) && this.fallbackAgent) {
        logger.warn('SDK', 'iFlow CLI API failed, falling back to Claude SDK', {
          sessionDbId: session.sessionDbId,
          error: error instanceof Error ? error.message : String(error),
          historyLength: session.conversationHistory.length
        });

        return this.fallbackAgent.startSession(session, worker);
      }

      logger.failure('SDK', 'iFlow CLI agent error', { sessionDbId: session.sessionDbId }, error as Error);
      throw error;
    } finally {
      // Ensure client is disconnected
      if (this.client) {
        try {
          await this.client.disconnect();
        } catch {
          // Ignore disconnect errors
        }
        this.client = null;
      }
    }
  }

  /**
   * Estimate token count from text (conservative estimate)
   */
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / CHARS_PER_TOKEN_ESTIMATE);
  }

  /**
   * Truncate conversation history to prevent runaway context costs
   */
  private truncateHistory(history: ConversationMessage[]): ConversationMessage[] {
    const settings = SettingsDefaultsManager.loadFromFile(path.join(homedir(), '.claude-mem', 'settings.json'));

    const MAX_CONTEXT_MESSAGES = parseInt(settings.CLAUDE_MEM_IFLOW_MAX_CONTEXT_MESSAGES) || DEFAULT_MAX_CONTEXT_MESSAGES;
    const MAX_ESTIMATED_TOKENS = parseInt(settings.CLAUDE_MEM_IFLOW_MAX_TOKENS) || DEFAULT_MAX_ESTIMATED_TOKENS;

    if (history.length <= MAX_CONTEXT_MESSAGES) {
      const totalTokens = history.reduce((sum, m) => sum + this.estimateTokens(m.content), 0);
      if (totalTokens <= MAX_ESTIMATED_TOKENS) {
        return history;
      }
    }

    // Sliding window: keep most recent messages within limits
    const truncated: ConversationMessage[] = [];
    let tokenCount = 0;

    for (let i = history.length - 1; i >= 0; i--) {
      const msg = history[i];
      const msgTokens = this.estimateTokens(msg.content);

      if (truncated.length >= MAX_CONTEXT_MESSAGES || tokenCount + msgTokens > MAX_ESTIMATED_TOKENS) {
        logger.warn('SDK', 'Context window truncated to prevent runaway costs', {
          originalMessages: history.length,
          keptMessages: truncated.length,
          droppedMessages: i + 1,
          estimatedTokens: tokenCount,
          tokenLimit: MAX_ESTIMATED_TOKENS
        });
        break;
      }

      truncated.unshift(msg);
      tokenCount += msgTokens;
    }

    return truncated;
  }

  /**
   * Query iFlow CLI via SDK with full conversation history (multi-turn)
   */
  private async queryiFlowMultiTurn(
    history: ConversationMessage[]
  ): Promise<{ content: string; tokensUsed?: number }> {
    if (!this.client) {
      throw new Error('iFlow client not initialized');
    }

    // Truncate history to prevent runaway costs
    const truncatedHistory = this.truncateHistory(history);
    const totalChars = truncatedHistory.reduce((sum, m) => sum + m.content.length, 0);
    const estimatedTokens = this.estimateTokens(truncatedHistory.map(m => m.content).join(''));

    // Build the prompt from conversation history
    // Last message is the current query, rest is context
    const lastMessage = truncatedHistory[truncatedHistory.length - 1];

    logger.debug('SDK', `Querying iFlow CLI multi-turn`, {
      turns: truncatedHistory.length,
      totalChars,
      estimatedTokens
    });

    try {
      // Send message using iFlow SDK
      await this.client.sendMessage(lastMessage.content);

      // Collect response
      let fullResponse = '';
      let tokenCount = 0;

      for await (const message of this.client.receiveMessages()) {
        if (message.type === MessageType.ASSISTANT && message.chunk?.text) {
          fullResponse += message.chunk.text;
        } else if (message.type === MessageType.TASK_FINISH) {
          break;
        } else if (message.type === MessageType.ERROR) {
          throw new Error(message.message || 'iFlow CLI API error');
        }
      }

      // Estimate tokens if not provided
      if (tokenCount === 0) {
        tokenCount = this.estimateTokens(fullResponse) + estimatedTokens;
      }

      return { content: fullResponse, tokensUsed: tokenCount };
    } catch (error) {
      logger.error('SDK', 'iFlow CLI query failed', {}, error as Error);
      throw error;
    }
  }
}

/**
 * Check if iFlow CLI is available (has valid authentication in ~/.iflow/settings.json)
 */
export function isiFlowCLIAvailable(): boolean {
  try {
    const settingsPath = path.join(homedir(), '.iflow', 'settings.json');
    if (!require('fs').existsSync(settingsPath)) {
      return false;
    }
    const settings = JSON.parse(require('fs').readFileSync(settingsPath, 'utf-8'));
    return !!(settings.accessToken || settings.apiKey);
  } catch {
    return false;
  }
}

/**
 * Check if iFlow CLI is the selected provider
 */
export function isiFlowCLISelected(): boolean {
  const settingsPath = path.join(homedir(), '.claude-mem', 'settings.json');
  const settings = SettingsDefaultsManager.loadFromFile(settingsPath);
  return settings.CLAUDE_MEM_PROVIDER === 'iflowcli';
}
