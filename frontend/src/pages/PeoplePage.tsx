import React from 'react';
import { 
  Grid, 
  Paper, 
  Typography, 
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';

const PeoplePage: React.FC = () => {
  const people = [
    {
      id: 'P001',
      firstSeen: '2024-01-14 10:30:00',
      lastSeen: '2024-01-14 10:45:00',
      zonesVisited: ['entrance', 'hallway', 'room-3'],
      status: 'active',
      confidence: 0.92,
    },
    {
      id: 'P002',
      firstSeen: '2024-01-14 09:15:00',
      lastSeen: '2024-01-14 09:30:00',
      zonesVisited: ['entrance', 'lobby'],
      status: 'exited',
      confidence: 0.88,
    },
  ];

  return (
    <Box>
      <Typography variant="h4" mb={3}>
        People Tracking
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Tracked Individuals
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>ID</TableCell>
                    <TableCell>First Seen</TableCell>
                    <TableCell>Last Seen</TableCell>
                    <TableCell>Zones Visited</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Confidence</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {people.map((person) => (
                    <TableRow key={person.id}>
                      <TableCell>{person.id}</TableCell>
                      <TableCell>{person.firstSeen}</TableCell>
                      <TableCell>{person.lastSeen}</TableCell>
                      <TableCell>
                        {person.zonesVisited.map((zone) => (
                          <Chip key={zone} label={zone} size="small" sx={{ mr: 0.5 }} />
                        ))}
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={person.status}
                          color={person.status === 'active' ? 'success' : 'default'}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>{Math.round(person.confidence * 100)}%</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default PeoplePage;