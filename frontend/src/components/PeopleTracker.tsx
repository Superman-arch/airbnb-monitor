import React from 'react';
import { Paper, Typography, Box, Avatar, AvatarGroup, Chip } from '@mui/material';
import { Person } from '@mui/icons-material';

interface TrackedPerson {
  id: string;
  confidence: number;
  firstSeen: string;
  lastSeen: string;
  zones: string[];
}

interface PeopleTrackerProps {
  people?: TrackedPerson[];
  currentCount?: number;
}

const PeopleTracker: React.FC<PeopleTrackerProps> = ({ 
  people = [], 
  currentCount = 0 
}) => {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        People Tracking
      </Typography>
      
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Box display="flex" alignItems="center">
          <Person sx={{ mr: 1 }} />
          <Typography variant="h4">{currentCount}</Typography>
          <Typography variant="body2" color="textSecondary" ml={1}>
            people detected
          </Typography>
        </Box>
        <AvatarGroup max={4}>
          {people.slice(0, 4).map((person) => (
            <Avatar key={person.id} sx={{ bgcolor: 'primary.main' }}>
              <Person />
            </Avatar>
          ))}
        </AvatarGroup>
      </Box>

      <Box>
        {people.slice(0, 5).map((person) => (
          <Box key={person.id} mb={1} p={1} sx={{ bgcolor: 'background.default', borderRadius: 1 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center">
              <Typography variant="body2">
                Person #{person.id.slice(-4)}
              </Typography>
              <Chip 
                label={`${Math.round(person.confidence * 100)}%`} 
                size="small" 
                color="primary"
              />
            </Box>
            <Typography variant="caption" color="textSecondary">
              Zones: {person.zones.join(', ')}
            </Typography>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default PeopleTracker;