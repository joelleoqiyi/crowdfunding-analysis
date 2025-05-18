
import React from "react";
import { AnalysisScore, AnalysisType } from "@/utils/types";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { FileText, Youtube, BarChart3, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface TabContentProps {
  type: AnalysisType;
  data: AnalysisScore;
}

const TabContent: React.FC<TabContentProps> = ({ type, data }) => {
  const getIcon = () => {
    switch (type) {
      case "description":
        return <FileText className="h-5 w-5" />;
      case "comments":
        return <BarChart3 className="h-5 w-5" />;
      case "youtube":
        return <Youtube className="h-5 w-5" />;
      case "updates":
        return <RefreshCw className="h-5 w-5" />;
    }
  };

  const getTitle = () => {
    switch (type) {
      case "description":
        return "Analysis by Description";
      case "comments":
        return "Analysis by Comments";
      case "youtube":
        return "Analysis by YouTube";
      case "updates":
        return "Analysis by Updates";
    }
  };

  const getScoreColor = () => {
    const { score } = data;
    if (score >= 70) return "text-analysis-success";
    if (score >= 50) return "text-analysis-warning";
    return "text-analysis-danger";
  };

  const getProgressColor = () => {
    const { score } = data;
    if (score >= 70) return "bg-analysis-success";
    if (score >= 50) return "bg-analysis-warning";
    return "bg-analysis-danger";
  };

  return (
    <div className="space-y-4 py-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-xl font-bold">
            {getIcon()}
            {getTitle()}
          </CardTitle>
        </CardHeader>
        { 
          type !== "description" ?
            <CardContent>
              <div className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium">Success Score</span>
                    <span className={`text-2xl font-bold ${getScoreColor()}`}>
                      {data.score}/100
                    </span>
                  </div>
                  <Progress value={data.score} className={getProgressColor()} />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Likely to Fail</span>
                    <span>Uncertain</span>
                    <span>Likely to Succeed</span>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Prediction</h4>
                  <div className="flex gap-2 items-center mb-4">
                    <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                      data.prediction === "success" 
                        ? "bg-analysis-success/20 text-analysis-success" 
                        : data.prediction === "failure" 
                        ? "bg-analysis-danger/20 text-analysis-danger" 
                        : "bg-analysis-warning/20 text-analysis-warning"
                    }`}>
                      {data.prediction === "success" 
                        ? "Likely to Succeed" 
                        : data.prediction === "failure" 
                        ? "Likely to Fail" 
                        : "Uncertain"
                      }
                    </span>
                    {type !== "comments" && <span className="text-sm text-muted-foreground">
                      Confidence: {Math.round(data.confidence * 100)}%
                    </span>}
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2">Justification</h4>
                  <p className="text-sm text-slate-600">{data.justification}</p>
                </div>

                <Separator />

                <div>
                  <h4 className="font-semibold mb-2">Key Findings</h4>
                  <ul className="space-y-1">
                    {data.findings.map((finding, index) => (
                      <li key={index} className="text-sm flex items-start gap-2">
                        <div className="h-1.5 w-1.5 rounded-full bg-primary mt-1.5" />
                        <span>{finding}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </CardContent>
          : // this is for descriptions only. link to the streamlit server
            <CardContent>
              <h4 className="font-semibold mb-2 mt-2 ml-1">StreamLit Dashboard</h4>
              <p className="text-sm text-slate-600 mb-2 ml-1">Kindly click on the button below to analyse the Campaign Description</p>
              <Button asChild className="ml-1">
                <a href="https://example.com" target="_blank">
                  Go to Dashboard
                </a>
              </Button>
            </CardContent>
        }
      </Card>
    </div>
  );
};

export default TabContent;
