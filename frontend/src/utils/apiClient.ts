import { AnalysisResults } from "./types";

// Default backend URL - updated to use port 5001
const API_BASE_URL = 'http://localhost:5001';

/**
 * Analyzes a Kickstarter campaign via URL
 * @param url The Kickstarter campaign URL to analyze
 * @returns Promise with analysis results
 */
export const analyzeCampaign = async (url: string): Promise<AnalysisResults> => {
  try {
    console.log(`Analyzing campaign at URL: ${url}`);
    
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Origin': window.location.origin,
      },
      body: JSON.stringify({ url }),
    });
    
    if (!response.ok) {
      console.error('Error response from API:', response.status, response.statusText);
      const errorText = await response.text();
      console.error('Error details:', errorText);
      throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log('API Response:', data);
    
    if (!data.success) {
      throw new Error(data.error || 'Unknown error occurred');
    }
    
    return data as AnalysisResults;
  } catch (error) {
    console.error('Error analyzing campaign:', error);
    throw error;
  }
};

/**
 * Checks if the backend API is available
 * @returns Promise<boolean> indicating if the API is available
 */
export const checkApiAvailability = async (): Promise<boolean> => {
  try {
    // Try a simple OPTIONS request to check CORS
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'OPTIONS',
      headers: {
        'Origin': window.location.origin,
      }
    });
    
    return response.ok;
  } catch (error) {
    console.error('API not available:', error);
    return false;
  }
};

/**
 * Fallback to dummy data when API is unavailable
 * This allows frontend testing without the backend
 */
export const getAnalysisWithFallback = async (url: string): Promise<AnalysisResults> => {
  try {
    // First check if API is available
    const isApiAvailable = await checkApiAvailability();
    
    if (isApiAvailable) {
      // Try to get real analysis
      return await analyzeCampaign(url);
    } else {
      console.warn('API not available, using dummy data');
      // Import dynamically to avoid circular dependencies
      const { generateDummyResults } = await import('./dummyData');
      return generateDummyResults(url);
    }
  } catch (error) {
    console.error('Error in analysis with fallback:', error);
    // Import dynamically to avoid circular dependencies
    const { generateDummyResults } = await import('./dummyData');
    return generateDummyResults(url);
  }
}; 