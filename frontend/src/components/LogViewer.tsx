import React from 'react';
import { Paper, Typography, Box, Chip, TextField, MenuItem } from '@mui/material';

interface LogEntry {
  id: string;
  level: 'debug' | 'info' | 'warning' | 'error';
  timestamp: string;
  message: string;
  source?: string;
}

interface LogViewerProps {
  logs?: LogEntry[];
  maxLines?: number;
}

const LogViewer: React.FC<LogViewerProps> = ({ logs = [], maxLines = 20 }) => {
  const [filter, setFilter] = React.useState<string>('all');

  const getLevelColor = (level: LogEntry['level']) => {
    switch (level) {
      case 'error':
        return 'error';
      case 'warning':
        return 'warning';
      case 'info':
        return 'info';
      default:
        return 'default';
    }
  };

  const filteredLogs = filter === 'all' 
    ? logs 
    : logs.filter(log => log.level === filter);

  return (
    <Paper sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant="h6">
          System Logs
        </Typography>
        <TextField
          select
          size="small"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          sx={{ minWidth: 120 }}
        >
          <MenuItem value="all">All</MenuItem>
          <MenuItem value="debug">Debug</MenuItem>
          <MenuItem value="info">Info</MenuItem>
          <MenuItem value="warning">Warning</MenuItem>
          <MenuItem value="error">Error</MenuItem>
        </TextField>
      </Box>
      
      <Box
        sx={{
          flexGrow: 1,
          overflow: 'auto',
          fontFamily: 'monospace',
          fontSize: '0.875rem',
          backgroundColor: 'background.default',
          p: 1,
          borderRadius: 1,
        }}
      >
        {filteredLogs.slice(0, maxLines).map((log) => (
          <Box key={log.id} mb={0.5}>
            <Box display="flex" alignItems="center" gap={1}>
              <Typography variant="caption" color="textSecondary">
                {log.timestamp}
              </Typography>
              <Chip
                label={log.level.toUpperCase()}
                size="small"
                color={getLevelColor(log.level)}
                sx={{ height: 16 }}
              />
              {log.source && (
                <Typography variant="caption" color="primary">
                  [{log.source}]
                </Typography>
              )}
            </Box>
            <Typography variant="body2" sx={{ ml: 2 }}>
              {log.message}
            </Typography>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default LogViewer;