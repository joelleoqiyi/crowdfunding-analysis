
import React, { useState } from "react";
import AnalysisInput from "@/components/AnalysisInput";
import AnalysisResults from "@/components/AnalysisResults";
import { AnalysisResults as AnalysisResultsType } from "@/utils/types";
import { generateDummyResults } from "@/utils/dummyData";
import { toast } from "sonner";

const Index: React.FC = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<AnalysisResultsType | null>(null);

  const handleAnalyze = async (url: string) => {
    setIsAnalyzing(true);
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2500));
      
      // In a real app, you would call your backend API here
      // const response = await fetch('your-api-endpoint', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ url })
      // });
      // const data = await response.json();
      
      // Using dummy data for now
      const dummyResults = generateDummyResults(url);
      setResults(dummyResults);
      toast.success("Analysis completed successfully!");
    } catch (error) {
      console.error("Analysis failed:", error);
      toast.error("Failed to analyze campaign. Please try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      <div className="container py-12 px-4 sm:px-6 lg:px-8">
        <header className="text-center mb-12">
          <h1 className="text-4xl font-bold text-slate-800 mb-3">
            Kickstarter Campaign Oracle
          </h1>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Leverage machine learning to predict your campaign's success. 
            Our AI analyzes descriptions, comments, videos, and updates to provide data-driven insights.
          </p>
        </header>

        <AnalysisInput onAnalyze={handleAnalyze} isLoading={isAnalyzing} />
        
        {isAnalyzing && (
          <div className="flex flex-col items-center justify-center mt-16">
            <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mb-4" />
            <p className="text-primary font-medium animate-pulse-soft">
              Analyzing campaign data...
            </p>
          </div>
        )}
        
        {!isAnalyzing && results && <AnalysisResults results={results} />}
        
        {!isAnalyzing && !results && (
          <div className="mt-16 text-center text-slate-500">
            <p>Enter a Kickstarter campaign URL above to begin analysis</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Index;
