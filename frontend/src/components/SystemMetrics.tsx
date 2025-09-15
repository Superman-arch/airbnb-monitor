import React from 'react';
import { Paper, Typography, Box, LinearProgress, Grid } from '@mui/material';
import { Speed, Memory, Storage, NetworkCheck } from '@mui/icons-material';

interface Metric {
  label: string;
  value: number;
  max: number;
  unit: string;
  icon: React.ReactNode;
}

interface SystemMetricsProps {
  fps?: number;
  cpuUsage?: number;
  memoryUsage?: number;
  gpuUsage?: number;
  diskUsage?: number;
  networkLatency?: number;
}

const SystemMetrics: React.FC<SystemMetricsProps> = ({
  fps = 30,
  cpuUsage = 45,
  memoryUsage = 60,
  gpuUsage = 75,
  diskUsage = 40,
  networkLatency = 15,
}) => {
  const metrics: Metric[] = [
    { label: 'FPS', value: fps, max: 60, unit: 'fps', icon: <Speed /> },
    { label: 'CPU', value: cpuUsage, max: 100, unit: '%', icon: <Memory /> },
    { label: 'Memory', value: memoryUsage, max: 100, unit: '%', icon: <Memory /> },
    { label: 'GPU', value: gpuUsage, max: 100, unit: '%', icon: <Speed /> },
    { label: 'Disk', value: diskUsage, max: 100, unit: '%', icon: <Storage /> },
    { label: 'Latency', value: networkLatency, max: 100, unit: 'ms', icon: <NetworkCheck /> },
  ];

  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        System Metrics
      </Typography>
      <Grid container spacing={2}>
        {metrics.map((metric) => (
          <Grid item xs={12} sm={6} key={metric.label}>
            <Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.5}>
                <Box display="flex" alignItems="center">
                  {metric.icon}
                  <Typography variant="body2" ml={1}>
                    {metric.label}
                  </Typography>
                </Box>
                <Typography variant="body2">
                  {metric.value}{metric.unit}
                </Typography>
              </Box>
              <LinearProgress
                variant="determinate"
                value={(metric.value / metric.max) * 100}
                color={
                  (metric.value / metric.max) > 0.8 ? 'error' : 
                  (metric.value / metric.max) > 0.6 ? 'warning' : 'primary'
                }
              />
            </Box>
          </Grid>
        ))}
      </Grid>
    </Paper>
  );
};

export default SystemMetrics;