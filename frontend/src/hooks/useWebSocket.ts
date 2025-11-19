/**
 * useWebSocket Hook
 * 
 * React hook for WebSocket connection with automatic reconnection
 * 
 * FIXED VERSION - Resolves all TypeScript errors
 */

import { useEffect, useRef, useCallback, useState } from 'react';
import { WebSocketEvent, WebSocketEventType } from '../types';

const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';

interface UseWebSocketOptions {
  videoId: string;
  onEvent?: (event: WebSocketEvent) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  reconnectInterval?: number;
  maxReconnectAttempts?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  lastEvent: WebSocketEvent | null;
  reconnectAttempts: number;
  send: (data: string) => void;
  disconnect: () => void;
}

export function useWebSocket({
  videoId,
  onEvent,
  onConnect,
  onDisconnect,
  onError,
  reconnectInterval = 3000,
  maxReconnectAttempts = 10,
}: UseWebSocketOptions): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WebSocketEvent | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const pingIntervalRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef(true);

  // ============================================================================
  // SEND MESSAGE
  // ============================================================================

  const send = useCallback((data: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(data);
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }, []);

  // ============================================================================
  // START PING INTERVAL (Keep-Alive)
  // ============================================================================

  const startPingInterval = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }

    pingIntervalRef.current = window.setInterval(() => {
      send('ping');
    }, 30000);
  }, [send]);

  // ============================================================================
  // STOP PING INTERVAL
  // ============================================================================

  const stopPingInterval = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }
  }, []);

  // ============================================================================
  // DISCONNECT
  // ============================================================================

  const disconnect = useCallback(() => {
    console.log('Manually disconnecting WebSocket');
    shouldReconnectRef.current = false;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    stopPingInterval();

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
  }, [stopPingInterval]);

  // ============================================================================
  // CONNECT TO WEBSOCKET
  // ============================================================================

  useEffect(() => {
    shouldReconnectRef.current = true;

    const connect = () => {
      // Don't reconnect if we've exceeded max attempts
      if (reconnectAttempts >= maxReconnectAttempts) {
        console.error('Max reconnection attempts reached');
        return;
      }

      // Close existing connection
      if (wsRef.current) {
        wsRef.current.close();
      }

      console.log(`Connecting to WebSocket: ${videoId}`);

      const ws = new WebSocket(`${WS_BASE_URL}/api/reviews/ws/${videoId}`);
      wsRef.current = ws;

      // ========== ON OPEN ==========
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setReconnectAttempts(0);
        startPingInterval();
        onConnect?.();
      };

      // ========== ON MESSAGE ==========
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as WebSocketEvent;
          console.log('WebSocket event received:', data.type);
          setLastEvent(data);
          onEvent?.(data);
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      // ========== ON CLOSE ==========
      ws.onclose = (event) => {
        console.log('WebSocket disconnected', event.code, event.reason);
        setIsConnected(false);
        stopPingInterval();
        onDisconnect?.();

        // Attempt reconnection if should reconnect
        if (shouldReconnectRef.current && reconnectAttempts < maxReconnectAttempts) {
          console.log(
            `Reconnecting in ${reconnectInterval}ms (attempt ${reconnectAttempts + 1}/${maxReconnectAttempts})`
          );

          reconnectTimeoutRef.current = window.setTimeout(() => {
            setReconnectAttempts((prev) => prev + 1);
          }, reconnectInterval);
        }
      };

      // ========== ON ERROR ==========
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        onError?.(error);
      };
    };

    connect();

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [
    videoId,
    reconnectAttempts,
    maxReconnectAttempts,
    reconnectInterval,
    onConnect,
    onDisconnect,
    onEvent,
    onError,
    startPingInterval,
    stopPingInterval,
    disconnect,
  ]);

  // ============================================================================
  // RETURN
  // ============================================================================

  return {
    isConnected,
    lastEvent,
    reconnectAttempts,
    send,
    disconnect,
  };
}

// ============================================================================
// TYPED EVENT HOOKS (Optional - for specific event types)
// ============================================================================

export function useWebSocketEvent<T extends WebSocketEvent>(
  videoId: string,
  eventType: WebSocketEventType,
  callback: (event: T) => void
) {
  const handleEvent = useCallback(
    (event: WebSocketEvent) => {
      if (event.type === eventType) {
        callback(event as T);
      }
    },
    [eventType, callback]
  );

  return useWebSocket({
    videoId,
    onEvent: handleEvent,
  });
}

export default useWebSocket;