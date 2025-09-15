import React, { useRef, useEffect } from 'react';
import { Box, Paper, Typography, CircularProgress } from '@mui/material';

interface VideoStreamProps {
  url?: string;
  title?: string;
}

const VideoStream: React.FC<VideoStreamProps> = ({ 
  url = '/api/stream', 
  title = 'Live Video Feed' 
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [loading, setLoading] = React.useState(true);

  useEffect(() => {
    // WebRTC or HLS setup would go here
    setLoading(false);
  }, [url]);

  return (
    <Paper sx={{ p: 2, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
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
        {loading ? (
          <Box
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          >
            <CircularProgress />
          </Box>
        ) : (
          <video
            ref={videoRef}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
            }}
            autoPlay
            muted
            playsInline
          />
        )}
      </Box>
    </Paper>
  );
};

export default VideoStream;