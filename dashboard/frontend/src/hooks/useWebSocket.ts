import { useEffect, useRef, useState, useCallback } from 'react';

// WebSocket connection states
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

// Message types for the dashboard
export interface WebSocketMessage {
  type: string;
  payload: unknown;
  timestamp?: string;
}

// Handler function type
export type MessageHandler = (payload: unknown) => void;

// Hook configuration options
export interface UseWebSocketOptions {
  url?: string;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
}

// Return type for the hook
export interface UseWebSocketReturn {
  status: ConnectionStatus;
  sendMessage: (message: WebSocketMessage) => void;
  registerHandler: (type: string, handler: MessageHandler) => void;
  unregisterHandler: (type: string) => void;
  reconnect: () => void;
  disconnect: () => void;
}

const DEFAULT_WS_URL = `ws://${typeof window !== 'undefined' ? window.location.hostname : 'localhost'}:8080/ws`;
const DEFAULT_RECONNECT_INTERVAL = 3000;
const DEFAULT_MAX_RECONNECT_ATTEMPTS = 10;

export function useWebSocket(options: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    url = DEFAULT_WS_URL,
    reconnectInterval = DEFAULT_RECONNECT_INTERVAL,
    maxReconnectAttempts = DEFAULT_MAX_RECONNECT_ATTEMPTS,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const wsRef = useRef<WebSocket | null>(null);
  const handlersRef = useRef<Map<string, MessageHandler>>(new Map());
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const shouldReconnectRef = useRef<boolean>(true);

  // Clear reconnect timeout
  const clearReconnectTimeout = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    // Clean up existing connection
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus('connecting');

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus('connected');
        reconnectAttemptsRef.current = 0;
        onConnect?.();
      };

      ws.onclose = () => {
        setStatus('disconnected');
        wsRef.current = null;
        onDisconnect?.();

        // Attempt reconnection if allowed
        if (shouldReconnectRef.current && reconnectAttemptsRef.current < maxReconnectAttempts) {
          setStatus('reconnecting');
          reconnectAttemptsRef.current += 1;

          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (error) => {
        onError?.(error);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          const { type, payload } = message;

          // Dispatch to registered handler
          const handler = handlersRef.current.get(type);
          if (handler) {
            handler(payload);
          }

          // Also dispatch to wildcard handler if registered
          const wildcardHandler = handlersRef.current.get('*');
          if (wildcardHandler) {
            wildcardHandler(message);
          }
        } catch (parseError) {
          console.error('Failed to parse WebSocket message:', parseError);
        }
      };
    } catch (connectionError) {
      console.error('Failed to create WebSocket connection:', connectionError);
      setStatus('disconnected');
    }
  }, [url, reconnectInterval, maxReconnectAttempts, onConnect, onDisconnect, onError]);

  // Send message through WebSocket
  const sendMessage = useCallback((message: WebSocketMessage) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      const messageWithTimestamp: WebSocketMessage = {
        ...message,
        timestamp: message.timestamp || new Date().toISOString(),
      };
      wsRef.current.send(JSON.stringify(messageWithTimestamp));
    } else {
      console.warn('WebSocket is not connected. Message not sent:', message);
    }
  }, []);

  // Register a message handler for a specific type
  const registerHandler = useCallback((type: string, handler: MessageHandler) => {
    handlersRef.current.set(type, handler);
  }, []);

  // Unregister a message handler
  const unregisterHandler = useCallback((type: string) => {
    handlersRef.current.delete(type);
  }, []);

  // Manual reconnect
  const reconnect = useCallback(() => {
    shouldReconnectRef.current = true;
    reconnectAttemptsRef.current = 0;
    clearReconnectTimeout();
    connect();
  }, [connect, clearReconnectTimeout]);

  // Disconnect and prevent auto-reconnect
  const disconnect = useCallback(() => {
    shouldReconnectRef.current = false;
    clearReconnectTimeout();

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus('disconnected');
  }, [clearReconnectTimeout]);

  // Connect on mount, disconnect on unmount
  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();

    return () => {
      shouldReconnectRef.current = false;
      clearReconnectTimeout();

      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [connect, clearReconnectTimeout]);

  return {
    status,
    sendMessage,
    registerHandler,
    unregisterHandler,
    reconnect,
    disconnect,
  };
}

export default useWebSocket;
