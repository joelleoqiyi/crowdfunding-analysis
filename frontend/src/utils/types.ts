
export type AnalysisType = 
  | "description" 
  | "comments" 
  | "youtube" 
  | "updates";

export interface AnalysisScore {
  score: number; // 0-100 score
  prediction: "success" | "failure" | "uncertain";
  confidence: number; // 0-1 confidence level
  justification: string;
  findings: string[];
}

export interface AnalysisResults {
  campaignTitle: string;
  campaignUrl: string;
  timestamp: string;
  description: AnalysisScore;
  comments: AnalysisScore;
  youtube: AnalysisScore;
  updates: AnalysisScore;
  overallPrediction: "success" | "failure" | "uncertain";
}
