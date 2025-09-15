import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  TextField,
  Switch,
  FormControlLabel,
  Button,
  Divider,
} from '@mui/material';
import { Save as SaveIcon } from '@mui/icons-material';

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = React.useState({
    webhookUrl: '',
    recordVideo: true,
    motionDetection: true,
    fpsLimit: 30,
    confidenceThreshold: 0.7,
    alertsEnabled: true,
    retentionDays: 7,
  });

  const handleSave = () => {
    console.log('Saving settings:', settings);
  };

  return (
    <Box>
      <Typography variant="h4" mb={3}>
        Settings
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              General Settings
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <TextField
                label="Webhook URL"
                value={settings.webhookUrl}
                onChange={(e) => setSettings({ ...settings, webhookUrl: e.target.value })}
                fullWidth
              />
              <TextField
                label="FPS Limit"
                type="number"
                value={settings.fpsLimit}
                onChange={(e) => setSettings({ ...settings, fpsLimit: parseInt(e.target.value) })}
                fullWidth
              />
              <TextField
                label="Retention Days"
                type="number"
                value={settings.retentionDays}
                onChange={(e) => setSettings({ ...settings, retentionDays: parseInt(e.target.value) })}
                fullWidth
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Detection Settings
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.motionDetection}
                    onChange={(e) => setSettings({ ...settings, motionDetection: e.target.checked })}
                  />
                }
                label="Motion Detection"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.recordVideo}
                    onChange={(e) => setSettings({ ...settings, recordVideo: e.target.checked })}
                  />
                }
                label="Record Video"
              />
              <FormControlLabel
                control={
                  <Switch
                    checked={settings.alertsEnabled}
                    onChange={(e) => setSettings({ ...settings, alertsEnabled: e.target.checked })}
                  />
                }
                label="Enable Alerts"
              />
              <TextField
                label="Confidence Threshold"
                type="number"
                value={settings.confidenceThreshold}
                onChange={(e) => setSettings({ ...settings, confidenceThreshold: parseFloat(e.target.value) })}
                inputProps={{ min: 0, max: 1, step: 0.1 }}
                fullWidth
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Box display="flex" justifyContent="flex-end">
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={handleSave}
              size="large"
            >
              Save Settings
            </Button>
          </Box>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SettingsPage;