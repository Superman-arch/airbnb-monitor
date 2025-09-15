import React from 'react';
import {
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Paper,
  Typography,
  Chip,
} from '@mui/material';
import { Warning, Info, Error, CheckCircle } from '@mui/icons-material';

interface Event {
  id: string;
  type: 'warning' | 'info' | 'error' | 'success';
  title: string;
  description: string;
  timestamp: string;
}

interface EventsListProps {
  events?: Event[];
  maxItems?: number;
}

const EventsList: React.FC<EventsListProps> = ({ 
  events = [], 
  maxItems = 10 
}) => {
  const getIcon = (type: Event['type']) => {
    switch (type) {
      case 'warning':
        return <Warning />;
      case 'error':
        return <Error />;
      case 'success':
        return <CheckCircle />;
      default:
        return <Info />;
    }
  };

  const getColor = (type: Event['type']) => {
    switch (type) {
      case 'warning':
        return 'warning';
      case 'error':
        return 'error';
      case 'success':
        return 'success';
      default:
        return 'info';
    }
  };

  const displayEvents = events.slice(0, maxItems);

  return (
    <Paper sx={{ p: 2, height: '100%', overflow: 'auto' }}>
      <Typography variant="h6" gutterBottom>
        Recent Events
      </Typography>
      <List>
        {displayEvents.length === 0 ? (
          <ListItem>
            <ListItemText 
              primary="No recent events" 
              secondary="System is operating normally"
            />
          </ListItem>
        ) : (
          displayEvents.map((event) => (
            <ListItem key={event.id} alignItems="flex-start">
              <ListItemAvatar>
                <Avatar sx={{ bgcolor: `${getColor(event.type)}.main` }}>
                  {getIcon(event.type)}
                </Avatar>
              </ListItemAvatar>
              <ListItemText
                primary={event.title}
                secondary={
                  <>
                    <Typography variant="body2" color="text.secondary">
                      {event.description}
                    </Typography>
                    <Chip 
                      label={event.timestamp} 
                      size="small" 
                      sx={{ mt: 0.5 }}
                    />
                  </>
                }
              />
            </ListItem>
          ))
        )}
      </List>
    </Paper>
  );
};

export default EventsList;