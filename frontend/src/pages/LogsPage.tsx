import React from 'react';
import { Box, Typography, Grid } from '@mui/material';
import LogViewer from '../components/LogViewer';

const LogsPage: React.FC = () => {
  const [logs] = React.useState([
    {
      id: '1',
      level: 'info' as const,
      timestamp: '2024-01-14 10:45:23',
      message: 'System started successfully',
      source: 'main',
    },
    {
      id: '2',
      level: 'warning' as const,
      timestamp: '2024-01-14 10:45:24',
      message: 'Door left open for more than 5 minutes',
      source: 'door-monitor',
    },
    {
      id: '3',
      level: 'debug' as const,
      timestamp: '2024-01-14 10:45:25',
      message: 'Person detected at entrance',
      source: 'detection',
    },
  ]);

  return (
    <Box>
      <Typography variant="h4" mb={3}>
        System Logs
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Box sx={{ height: 'calc(100vh - 200px)' }}>
            <LogViewer logs={logs} maxLines={100} />
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default LogsPage;