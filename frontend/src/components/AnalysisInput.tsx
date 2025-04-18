
import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";

interface AnalysisInputProps {
  onAnalyze: (url: string) => void;
  isLoading: boolean;
}

const AnalysisInput: React.FC<AnalysisInputProps> = ({ onAnalyze, isLoading }) => {
  const [url, setUrl] = useState("");
  
  const validateUrl = (url: string): boolean => {
    return url.trim() !== "" && 
           (url.includes("kickstarter.com") || 
            url.includes("indiegogo.com"));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateUrl(url)) {
      toast.error("Please enter a valid Kickstarter or Indiegogo campaign URL");
      return;
    }
    
    onAnalyze(url);
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <div className="bg-white/5 backdrop-blur-md shadow-xl rounded-lg p-6 border border-slate-200/20">
        <h2 className="text-xl font-semibold mb-4 text-slate-700">Enter Campaign URL</h2>
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input
            type="url"
            placeholder="https://www.kickstarter.com/projects/example/campaign-name"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            className="flex-1"
            disabled={isLoading}
            required
          />
          <Button type="submit" disabled={isLoading}>
            {isLoading ? (
              <span className="flex items-center gap-2">
                <span className="h-4 w-4 rounded-full bg-white/80 animate-pulse" />
                Analyzing...
              </span>
            ) : (
              "Analyze Campaign"
            )}
          </Button>
        </form>
      </div>
    </div>
  );
};

export default AnalysisInput;
