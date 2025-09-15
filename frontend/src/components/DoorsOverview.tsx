import React from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
} from '@mui/material';
import { DoorFront, Lock, LockOpen } from '@mui/icons-material';

interface Door {
  id: string;
  name: string;
  state: 'open' | 'closed' | 'unknown';
  lastChange: string;
  confidence: number;
}

interface DoorsOverviewProps {
  doors?: Door[];
}

const DoorsOverview: React.FC<DoorsOverviewProps> = ({ doors = [] }) => {
  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Doors Status
      </Typography>
      <Grid container spacing={2}>
        {doors.length === 0 ? (
          <Grid item xs={12}>
            <Card>
              <CardContent>
                <Typography color="textSecondary">
                  No doors configured
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        ) : (
          doors.map((door) => (
            <Grid item xs={12} sm={6} md={4} key={door.id}>
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="center">
                    <Box display="flex" alignItems="center">
                      <DoorFront sx={{ mr: 1 }} />
                      <Typography variant="h6">
                        {door.name}
                      </Typography>
                    </Box>
                    {door.state === 'open' ? (
                      <LockOpen color="warning" />
                    ) : (
                      <Lock color="success" />
                    )}
                  </Box>
                  <Box mt={2}>
                    <Chip
                      label={door.state.toUpperCase()}
                      color={door.state === 'open' ? 'warning' : 'success'}
                      size="small"
                    />
                    <Typography variant="caption" display="block" mt={1}>
                      Last change: {door.lastChange}
                    </Typography>
                    <Box mt={1}>
                      <Typography variant="caption">
                        Confidence: {Math.round(door.confidence * 100)}%
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={door.confidence * 100}
                        sx={{ mt: 0.5 }}
                      />
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))
        )}
      </Grid>
    </Box>
  );
};

export default DoorsOverview;