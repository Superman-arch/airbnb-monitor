import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Box, Paper, Typography, CircularProgress, Alert, IconButton, Chip } from '@mui/material';
import { Refresh as RefreshIcon, VideocamOff, Videocam } from '@mui/icons-material';

interface VideoStreamWebSocketProps {
  title?: string;
  useWebSocket?: boolean;
}

const VideoStreamWebSocket: React.FC<VideoStreamWebSocketProps> = ({ 
  title = 'Live Video Feed',
  useWebSocket = false
}) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const [connectionType, setConnectionType] = useState<'mjpeg' | 'websocket'>('mjpeg');
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  const connectToMJPEGStream = useCallback(() => {
    if (imgRef.current) {
      setError(null);
      setLoading(true);
      setConnectionType('mjpeg');
      
      // Add timestamp to force reload
      const timestamp = new Date().getTime();
      imgRef.current.src = `/api/streams/video/live?t=${timestamp}`;
    }
  }, []);

  const connectToWebSocket = useCallback(() => {
    setError(null);
    setLoading(true);
    setConnectionType('websocket');

    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      // Determine WebSocket URL based on current location
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.hostname;
      const port = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
      const wsUrl = `${protocol}//${host}:${port}/api/streams/ws/video`;
      
      console.log('Connecting to WebSocket:', wsUrl);
      
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setLoading(false);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          if (message.type === 'frame' && message.data) {
            // Decode base64 frame data
            const imageData = atob(message.data);
            const arrayBuffer = new ArrayBuffer(imageData.length);
            const uint8Array = new Uint8Array(arrayBuffer);
            
            for (let i = 0; i < imageData.length; i++) {
              uint8Array[i] = imageData.charCodeAt(i);
            }
            
            // Create blob and display
            const blob = new Blob([uint8Array], { type: 'image/jpeg' });
            const imageUrl = URL.createObjectURL(blob);
            
            // Draw to canvas
            if (canvasRef.current) {
              const img = new Image();
              img.onload = () => {
                const ctx = canvasRef.current?.getContext('2d');
                if (ctx && canvasRef.current) {
                  canvasRef.current.width = img.width;
                  canvasRef.current.height = img.height;
                  ctx.drawImage(img, 0, 0);
                  URL.revokeObjectURL(imageUrl);
                  
                  // Update FPS
                  frameCountRef.current++;
                  const now = Date.now();
                  if (now - lastFpsUpdateRef.current >= 1000) {
                    setFps(frameCountRef.current);
                    frameCountRef.current = 0;
                    lastFpsUpdateRef.current = now;
                  }
                }
              };
              img.src = imageUrl;
            }
          } else if (message.type === 'connection') {
            console.log('Connection message:', message);
          }
        } catch (e) {
          console.error('Error processing WebSocket message:', e);
        }
      };

      ws.onerror = (event) => {
        console.error('WebSocket error:', event);
        setError('WebSocket connection error');
        setLoading(false);
      };

      ws.onclose = () => {
        console.log('WebSocket closed');
        setError('WebSocket connection closed');
        setLoading(false);
        
        // Auto-retry after 5 seconds
        setTimeout(() => {
          setRetryCount(prev => prev + 1);
        }, 5000);
      };
    } catch (e) {
      console.error('Failed to connect WebSocket:', e);
      setError('Failed to connect to video stream');
      setLoading(false);
    }
  }, []);

  const connectToStream = useCallback(() => {
    if (useWebSocket || connectionType === 'websocket') {
      connectToWebSocket();
    } else {
      connectToMJPEGStream();
    }
  }, [useWebSocket, connectionType, connectToWebSocket, connectToMJPEGStream]);

  useEffect(() => {
    connectToStream();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [retryCount]);

  const handleImageLoad = () => {
    setLoading(false);
    setError(null);
  };

  const handleImageError = () => {
    setLoading(false);
    
    if (connectionType === 'mjpeg') {
      // Try WebSocket as fallback
      console.log('MJPEG failed, trying WebSocket...');
      setConnectionType('websocket');
      setRetryCount(prev => prev + 1);
    } else {
      setError('Unable to connect to video stream');
      
      // Auto-retry after 5 seconds
      setTimeout(() => {
        setRetryCount(prev => prev + 1);
      }, 5000);
    }
  };

  const handleManualRetry = () => {
    setConnectionType('mjpeg'); // Start with MJPEG again
    setRetryCount(prev => prev + 1);
  };

  const toggleConnectionType = () => {
    setConnectionType(prev => prev === 'mjpeg' ? 'websocket' : 'mjpeg');
    setRetryCount(prev => prev + 1);
  };

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography variant="h6">
            {title}
          </Typography>
          <Chip 
            size="small" 
            label={connectionType.toUpperCase()} 
            color={connectionType === 'websocket' ? 'primary' : 'default'}
          />
          {connectionType === 'websocket' && fps > 0 && (
            <Chip size="small" label={`${fps} FPS`} color="success" />
          )}
        </Box>
        <Box>
          <IconButton onClick={toggleConnectionType} size="small" title="Toggle connection type">
            {connectionType === 'websocket' ? <Videocam /> : <VideocamOff />}
          </IconButton>
          <IconButton onClick={handleManualRetry} size="small" disabled={loading}>
            <RefreshIcon />
          </IconButton>
        </Box>
      </Box>
      
      <Box
        sx={{
          position: 'relative',
          width: '100%',
          paddingTop: '56.25%', // 16:9 aspect ratio
          backgroundColor: 'black',
          borderRadius: 1,
          overflow: 'hidden',
        }}
      >
        {loading && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 2,
            }}
          >
            <CircularProgress />
          </Box>
        )}
        
        {error && (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              zIndex: 2,
              width: '80%',
              textAlign: 'center',
            }}
          >
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
            <Typography variant="body2" color="text.secondary">
              Retrying automatically...
            </Typography>
          </Box>
        )}
        
        {connectionType === 'mjpeg' ? (
          <img
            ref={imgRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              display: loading || error ? 'none' : 'block',
            }}
            onLoad={handleImageLoad}
            onError={handleImageError}
            alt="Video Stream"
          />
        ) : (
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              display: loading || error ? 'none' : 'block',
            }}
          />
        )}
      </Box>
    </Paper>
  );
};

export default VideoStreamWebSocket;