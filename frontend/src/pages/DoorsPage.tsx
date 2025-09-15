import React from 'react';
import { Grid, Paper, Typography, Button, Box } from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import DoorsOverview from '../components/DoorsOverview';

const DoorsPage: React.FC = () => {
  const [doors] = React.useState([
    {
      id: '1',
      name: 'Main Entrance',
      state: 'closed' as const,
      lastChange: '5 mins ago',
      confidence: 0.95,
    },
    {
      id: '2',
      name: 'Back Door',
      state: 'open' as const,
      lastChange: '2 hours ago',
      confidence: 0.88,
    },
  ]);

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4">
          Door Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => console.log('Add door')}
        >
          Add Door
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <DoorsOverview doors={doors} />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default DoorsPage;