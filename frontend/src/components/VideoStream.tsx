import React, { useRef, useEffect, useState } from 'react';
import { Box, Paper, Typography, CircularProgress, Alert, IconButton } from '@mui/material';
import { Refresh as RefreshIcon } from '@mui/icons-material';

interface VideoStreamProps {
  url?: string;
  title?: string;
}

const VideoStream: React.FC<VideoStreamProps> = ({ 
  url = '/api/streams/video/live', 
  title = 'Live Video Feed' 
}) => {
  const imgRef = useRef<HTMLImageElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);

  const connectToStream = () => {
    if (imgRef.current) {
      setError(null);
      setLoading(true);
      
      // Add timestamp to force reload
      const timestamp = new Date().getTime();
      imgRef.current.src = `${url}?t=${timestamp}`;
    }
  };

  useEffect(() => {
    connectToStream();
  }, [url, retryCount]);

  const handleImageLoad = () => {
    setLoading(false);
    setError(null);
  };

  const handleImageError = () => {
    setLoading(false);
    setError('Unable to connect to video stream');
    
    // Auto-retry after 5 seconds
    setTimeout(() => {
      setRetryCount(prev => prev + 1);
    }, 5000);
  };

  const handleManualRetry = () => {
    setRetryCount(prev => prev + 1);
  };

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6">
          {title}
        </Typography>
        <IconButton onClick={handleManualRetry} size="small" disabled={loading}>
          <RefreshIcon />
        </IconButton>
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
      </Box>
    </Paper>
  );
};

export default VideoStream;