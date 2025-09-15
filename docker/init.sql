-- Initialize database schema for security monitoring system

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create initial tables will be handled by SQLAlchemy/Alembic
-- This file is for any custom initialization needed

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE security_monitoring TO security;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO security;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO security;