import { useEffect, useState, useCallback, useRef } from 'react';

interface WebSocketData {
  stats: {
    fps: number;
    peopleCount: number;
    doorsOpen: number;
    uptime: string;
  };
  events: Array<{
    id: string;
    type: string;
    message: string;
    timestamp: string;
  }>;
  doors: Array<{
    id: string;
    name: string;
    state: string;
    confidence: number;
  }>;
  people: Array<{
    id: string;
    zones: string[];
    confidence: number;
  }>;
  isConnected: boolean;
}

export const useWebSocket = (path: string = '/ws') => {
  const [data, setData] = useState<WebSocketData>({
    stats: {
      fps: 0,
      peopleCount: 0,
      doorsOpen: 0,
      uptime: '00:00:00',
    },
    events: [],
    doors: [],
    people: [],
    isConnected: false,
  });
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const reconnectAttemptsRef = useRef(0);

  const connect = useCallback(() => {
    try {
      // Build WebSocket URL based on current location
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}${path}`;
      
      console.log('Connecting to WebSocket:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setData(prev => ({ ...prev, isConnected: true }));
        reconnectAttemptsRef.current = 0;
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setData(prev => ({ ...prev, isConnected: false }));
        wsRef.current = null;
        
        // Attempt to reconnect
        if (reconnectAttemptsRef.current < 10) {
          const delay = Math.min(1000 * Math.pow(2, reconnectAttemptsRef.current), 30000);
          reconnectAttemptsRef.current++;
          
          console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`);
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, delay);
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
      
      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'connection':
              console.log('Connection message:', message);
              break;
              
            case 'stats':
              setData(prev => ({ 
                ...prev, 
                stats: {
                  fps: message.data.fps || 0,
                  peopleCount: message.data.peopleCount || 0,
                  doorsOpen: message.data.doorsOpen || 0,
                  uptime: message.data.uptime || '00:00:00',
                }
              }));
              break;
              
            case 'event':
              setData(prev => ({
                ...prev,
                events: [message.data, ...prev.events].slice(0, 100),
              }));
              break;
              
            case 'doors_update':
              setData(prev => ({ ...prev, doors: message.data }));
              break;
              
            case 'people_update':
              setData(prev => ({ ...prev, people: message.data }));
              break;
              
            default:
              console.log('Unknown message type:', message.type);
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setData(prev => ({ ...prev, isConnected: false }));
    }
  }, [path]);

  useEffect(() => {
    connect();
    
    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  const sendMessage = useCallback((type: string, data: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, data }));
    } else {
      console.warn('WebSocket not connected');
    }
  }, []);

  return { ...data, sendMessage };
};