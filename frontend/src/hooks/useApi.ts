import { useState, useEffect, useCallback } from 'react';
import axios, { AxiosError } from 'axios';

// Configure axios defaults
axios.defaults.baseURL = '';  // Use relative URLs

interface UseApiOptions {
  autoFetch?: boolean;
  pollingInterval?: number;
}

export function useApi<T = any>(
  url: string,
  options: UseApiOptions = { autoFetch: true }
) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios.get<T>(url);
      setData(response.data);
    } catch (err) {
      const error = err as AxiosError;
      console.error(`API Error fetching ${url}:`, error.response?.data || error.message);
      const errorData = error.response?.data as any;
      setError(new Error(errorData?.detail || error.message || 'Failed to fetch data'));
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    if (options.autoFetch) {
      fetchData();
    }

    if (options.pollingInterval) {
      const interval = setInterval(fetchData, options.pollingInterval);
      return () => clearInterval(interval);
    }
  }, [fetchData, options.autoFetch, options.pollingInterval]);

  const refetch = useCallback(() => {
    return fetchData();
  }, [fetchData]);

  const mutate = useCallback(async (method: 'post' | 'put' | 'patch' | 'delete', payload?: any) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await axios[method]<T>(url, payload);
      setData(response.data);
      return response.data;
    } catch (err) {
      const error = err as AxiosError;
      setError(new Error(error.message || 'Failed to mutate data'));
      throw error;
    } finally {
      setLoading(false);
    }
  }, [url]);

  return {
    data,
    loading,
    error,
    refetch,
    mutate,
  };
}