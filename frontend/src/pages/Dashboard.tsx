import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Paper,
  Chip,
  IconButton,
  Button,
  Tooltip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  DoorFront,
  Person,
  Speed,
  Memory,
  Storage,
  Videocam,
  PlayArrow,
  Pause,
  FiberManualRecord,
  Fullscreen,
  ZoomIn,
  ZoomOut,
  CameraAlt,
  Warning,
  Error,
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

// Components
import VideoStream from '../components/VideoStream';
import VideoStreamWebSocket from '../components/VideoStreamWebSocket';
import StatsCard from '../components/StatsCard';
import EventsList from '../components/EventsList';
import DoorsOverview from '../components/DoorsOverview';
import SystemMetrics from '../components/SystemMetrics';
import LogViewer from '../components/LogViewer';

// Hooks
import { useWebSocket } from '../hooks/useWebSocket';
import { useApi } from '../hooks/useApi';

// Stores
import { useMonitoringStore } from '../stores/monitoringStore';

const Dashboard: React.FC = () => {
  const { stats, events, doors, people, isConnected } = useWebSocket();
  const { data: systemStatus } = useApi('/api/health');
  const { isRecording, toggleRecording, isPaused, togglePause } = useMonitoringStore();

  const [videoZoom, setVideoZoom] = useState(1);
  const [showOverlays, setShowOverlays] = useState(true);
  const [alertsExpanded, setAlertsExpanded] = useState(false);

  // Calculate real-time metrics
  const metrics = {
    fps: stats?.fps || 0,
    latency: 0, // Add default value
    doorsOpen: doors?.filter(d => d.state === 'open').length || 0,
    doorsClosed: doors?.filter(d => d.state === 'closed').length || 0,
    peopleCount: people?.length || 0,
    memoryUsage: 0, // Add default value
    gpuUsage: 0, // Add default value
  };

  // Alert calculations
  const alerts = {
    doorsLeftOpen: doors?.filter(d => 
      d.state === 'open'
    ).length || 0,
    crowding: people?.length > 20,
    systemHealth: systemStatus?.status === 'healthy',
  };

  return (
    <Box sx={{ flexGrow: 1, p: 2 }}>
      {/* Connection Status Bar */}
      <Paper 
        sx={{ 
          mb: 2, 
          p: 1, 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'space-between',
          background: isConnected 
            ? 'linear-gradient(90deg, #4CAF50 0%, #45a049 100%)' 
            : 'linear-gradient(90deg, #f44336 0%, #da190b 100%)',
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <FiberManualRecord 
            sx={{ 
              fontSize: 12, 
              color: isConnected ? '#fff' : '#ffcdd2',
              animation: isConnected ? 'pulse 2s infinite' : 'none'
            }} 
          />
          <Typography variant="body2" sx={{ color: '#fff', fontWeight: 600 }}>
            {isConnected ? 'SYSTEM ONLINE' : 'CONNECTION LOST'}
          </Typography>
          {isConnected && (
            <>
              <Chip 
                label={`FPS: ${metrics.fps.toFixed(1)}`} 
                size="small" 
                sx={{ 
                  ml: 2, 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: '#fff',
                  fontWeight: 600 
                }} 
              />
              <Chip 
                label={`Latency: ${metrics.latency}ms`} 
                size="small" 
                sx={{ 
                  backgroundColor: 'rgba(255,255,255,0.2)', 
                  color: '#fff',
                  fontWeight: 600 
                }} 
              />
            </>
          )}
        </Box>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          {/* Recording Controls */}
          <Tooltip title={isRecording ? "Stop Recording" : "Start Recording"}>
            <IconButton 
              size="small" 
              onClick={toggleRecording}
              sx={{ color: '#fff' }}
            >
              {isRecording ? (
                <FiberManualRecord sx={{ color: '#ff5252' }} />
              ) : (
                <FiberManualRecord />
              )}
            </IconButton>
          </Tooltip>
          
          <Tooltip title={isPaused ? "Resume" : "Pause"}>
            <IconButton 
              size="small" 
              onClick={togglePause}
              sx={{ color: '#fff' }}
            >
              {isPaused ? <PlayArrow /> : <Pause />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Take Snapshot">
            <IconButton size="small" sx={{ color: '#fff' }}>
              <CameraAlt />
            </IconButton>
          </Tooltip>
        </Box>
      </Paper>

      {/* Alerts Section */}
      {(alerts.doorsLeftOpen > 0 || alerts.crowding || !alerts.systemHealth) && (
        <AnimatePresence>
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <Alert 
              severity={alerts.doorsLeftOpen > 0 || alerts.crowding ? "warning" : "error"}
              sx={{ mb: 2 }}
              action={
                <Button 
                  size="small" 
                  onClick={() => setAlertsExpanded(!alertsExpanded)}
                >
                  {alertsExpanded ? 'Hide' : 'Details'}
                </Button>
              }
            >
              <Box>
                {alerts.doorsLeftOpen > 0 && (
                  <Typography variant="body2">
                    <Warning sx={{ fontSize: 16, verticalAlign: 'middle', mr: 1 }} />
                    {alerts.doorsLeftOpen} door(s) left open for more than 5 minutes
                  </Typography>
                )}
                {alerts.crowding && (
                  <Typography variant="body2">
                    <Warning sx={{ fontSize: 16, verticalAlign: 'middle', mr: 1 }} />
                    High occupancy detected ({metrics.peopleCount} people)
                  </Typography>
                )}
                {!alerts.systemHealth && (
                  <Typography variant="body2">
                    <Error sx={{ fontSize: 16, verticalAlign: 'middle', mr: 1 }} />
                    System health check failed
                  </Typography>
                )}
              </Box>
            </Alert>
          </motion.div>
        </AnimatePresence>
      )}

      <Grid container spacing={2}>
        {/* Main Video Feed with Overlays */}
        <Grid item xs={12} lg={8}>
          <Card sx={{ height: '600px', position: 'relative' }}>
            <CardContent sx={{ p: 0, height: '100%', position: 'relative' }}>
              <VideoStreamWebSocket />
              
              {/* Video Controls Overlay */}
              <Box
                sx={{
                  position: 'absolute',
                  bottom: 16,
                  left: 16,
                  right: 16,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  backgroundColor: 'rgba(0,0,0,0.7)',
                  borderRadius: 2,
                  p: 1,
                  backdropFilter: 'blur(10px)',
                }}
              >
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Tooltip title="Toggle Overlays">
                    <IconButton 
                      size="small" 
                      onClick={() => setShowOverlays(!showOverlays)}
                      sx={{ color: showOverlays ? '#4CAF50' : '#fff' }}
                    >
                      <Videocam />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Zoom In">
                    <IconButton 
                      size="small" 
                      onClick={() => setVideoZoom(Math.min(3, videoZoom + 0.5))}
                      sx={{ color: '#fff' }}
                    >
                      <ZoomIn />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Zoom Out">
                    <IconButton 
                      size="small" 
                      onClick={() => setVideoZoom(Math.max(1, videoZoom - 0.5))}
                      sx={{ color: '#fff' }}
                    >
                      <ZoomOut />
                    </IconButton>
                  </Tooltip>
                  
                  <Tooltip title="Fullscreen">
                    <IconButton size="small" sx={{ color: '#fff' }}>
                      <Fullscreen />
                    </IconButton>
                  </Tooltip>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <Chip 
                    icon={<DoorFront />} 
                    label={`Doors: ${metrics.doorsOpen}/${metrics.doorsOpen + metrics.doorsClosed}`}
                    size="small"
                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: '#fff' }}
                  />
                  <Chip 
                    icon={<Person />} 
                    label={`People: ${metrics.peopleCount}`}
                    size="small"
                    sx={{ backgroundColor: 'rgba(255,255,255,0.2)', color: '#fff' }}
                  />
                </Box>
              </Box>

              {/* Performance Overlay */}
              <Box
                sx={{
                  position: 'absolute',
                  top: 16,
                  left: 16,
                  backgroundColor: 'rgba(0,0,0,0.8)',
                  borderRadius: 1,
                  p: 1.5,
                  backdropFilter: 'blur(10px)',
                  minWidth: 200,
                }}
              >
                <Typography variant="caption" sx={{ color: '#4CAF50', fontWeight: 600, display: 'block', mb: 1 }}>
                  SYSTEM PERFORMANCE
                </Typography>
                
                <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                  <Speed sx={{ fontSize: 14, mr: 1, color: '#fff' }} />
                  <Typography variant="caption" sx={{ color: '#aaa', mr: 1 }}>FPS:</Typography>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: metrics.fps >= 15 ? '#4CAF50' : metrics.fps >= 10 ? '#ff9800' : '#f44336',
                      fontWeight: 600 
                    }}
                  >
                    {metrics.fps.toFixed(1)}
                  </Typography>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                  <Memory sx={{ fontSize: 14, mr: 1, color: '#fff' }} />
                  <Typography variant="caption" sx={{ color: '#aaa', mr: 1 }}>RAM:</Typography>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: metrics.memoryUsage < 60 ? '#4CAF50' : metrics.memoryUsage < 80 ? '#ff9800' : '#f44336',
                      fontWeight: 600 
                    }}
                  >
                    {metrics.memoryUsage.toFixed(0)}%
                  </Typography>
                </Box>

                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <Storage sx={{ fontSize: 14, mr: 1, color: '#fff' }} />
                  <Typography variant="caption" sx={{ color: '#aaa', mr: 1 }}>GPU:</Typography>
                  <Typography 
                    variant="caption" 
                    sx={{ 
                      color: metrics.gpuUsage < 70 ? '#4CAF50' : metrics.gpuUsage < 90 ? '#ff9800' : '#f44336',
                      fontWeight: 600 
                    }}
                  >
                    {metrics.gpuUsage.toFixed(0)}%
                  </Typography>
                </Box>

                <LinearProgress 
                  variant="determinate" 
                  value={metrics.fps * 100 / 30} 
                  sx={{ mt: 1, height: 2 }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Stats and Controls Panel */}
        <Grid item xs={12} lg={4}>
          <Grid container spacing={2}>
            {/* Quick Stats */}
            <Grid item xs={6}>
              <StatsCard
                title="Active Doors"
                value={metrics.doorsOpen}
                icon={<DoorFront />}
                color="#4CAF50"
              />
            </Grid>
            
            <Grid item xs={6}>
              <StatsCard
                title="People Count"
                value={metrics.peopleCount}
                icon={<Person />}
                color="#2196F3"
              />
            </Grid>

            {/* Doors Overview */}
            <Grid item xs={12}>
              <DoorsOverview 
                doors={(doors || []).map(d => ({
                  ...d,
                  state: d.state as 'open' | 'closed' | 'unknown',
                  lastChange: '5 mins ago'
                }))}
              />
            </Grid>

            {/* Events List */}
            <Grid item xs={12}>
              <EventsList 
                events={(events || []).map(e => ({
                  ...e,
                  type: e.type as 'warning' | 'info' | 'error' | 'success',
                  title: e.message || 'Event',
                  description: e.timestamp || 'No details'
                }))
                }
                maxItems={10}
              />
            </Grid>
          </Grid>
        </Grid>

        {/* Bottom Section - Logs and Metrics */}
        <Grid item xs={12} lg={6}>
          <LogViewer />
        </Grid>

        <Grid item xs={12} lg={6}>
          <SystemMetrics />
        </Grid>
      </Grid>

      <style>{`
        @keyframes pulse {
          0% {
            opacity: 1;
          }
          50% {
            opacity: 0.5;
          }
          100% {
            opacity: 1;
          }
        }
      `}</style>
    </Box>
  );
};

export default Dashboard;