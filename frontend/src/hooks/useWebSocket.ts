import { useEffect, useState, useCallback } from 'react';
import io, { Socket } from 'socket.io-client';

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

export const useWebSocket = (url: string = '/ws') => {
  const [socket, setSocket] = useState<Socket | null>(null);
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

  useEffect(() => {
    const newSocket = io(url, {
      path: '/ws/socket.io/',
      transports: ['websocket', 'polling'],
    });

    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setData(prev => ({ ...prev, isConnected: true }));
    });

    newSocket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      setData(prev => ({ ...prev, isConnected: false }));
    });

    newSocket.on('stats_update', (stats: any) => {
      setData(prev => ({ ...prev, stats }));
    });

    newSocket.on('event', (event: any) => {
      setData(prev => ({
        ...prev,
        events: [event, ...prev.events].slice(0, 100),
      }));
    });

    newSocket.on('doors_update', (doors: any) => {
      setData(prev => ({ ...prev, doors }));
    });

    newSocket.on('people_update', (people: any) => {
      setData(prev => ({ ...prev, people }));
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, [url]);

  const sendMessage = useCallback((event: string, data: any) => {
    if (socket) {
      socket.emit(event, data);
    }
  }, [socket]);

  return { ...data, sendMessage };
};