
import React, { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { AnalysisResults as AnalysisResultsType } from "@/utils/types";
import { Badge } from "@/components/ui/badge";
import TabContent from "./TabContent";
import { FileText, BarChart3, Youtube, RefreshCw } from "lucide-react";

interface AnalysisResultsProps {
  results: AnalysisResultsType;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ results }) => {
  const [activeTab, setActiveTab] = useState("description");
  
  return (
    <div className="w-full max-w-5xl mx-auto mt-8">
      <Card className="border border-slate-200 shadow-md">
        <CardHeader>
          <div className="flex justify-between items-start">
            <div>
              <CardTitle className="text-2xl">{results.campaignTitle}</CardTitle>
              <CardDescription className="text-sm truncate max-w-md">
                <a href={results.campaignUrl} target="_blank" rel="noopener noreferrer" className="hover:underline">
                  {results.campaignUrl}
                </a>
              </CardDescription>
            </div>
            <Badge className={
              results.overallPrediction === "success" 
                ? "bg-analysis-success" 
                : results.overallPrediction === "failure" 
                ? "bg-analysis-danger" 
                : "bg-analysis-warning"
            }>
              {results.overallPrediction === "success" 
                ? "Likely to Succeed" 
                : results.overallPrediction === "failure" 
                ? "Likely to Fail" 
                : "Prediction Uncertain"
              }
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="description" value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="grid grid-cols-4 mb-6">
              <TabsTrigger value="description" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                <span className="hidden sm:inline">Description</span>
              </TabsTrigger>
              <TabsTrigger value="comments" className="flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                <span className="hidden sm:inline">Comments</span>
              </TabsTrigger>
              <TabsTrigger value="youtube" className="flex items-center gap-2">
                <Youtube className="h-4 w-4" />
                <span className="hidden sm:inline">YouTube</span>
              </TabsTrigger>
              <TabsTrigger value="updates" className="flex items-center gap-2">
                <RefreshCw className="h-4 w-4" />
                <span className="hidden sm:inline">Updates</span>
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="description">
              <TabContent type="description" data={results.description} />
            </TabsContent>
            
            <TabsContent value="comments">
              <TabContent type="comments" data={results.comments} />
            </TabsContent>
            
            <TabsContent value="youtube">
              <TabContent type="youtube" data={results.youtube} />
            </TabsContent>
            
            <TabsContent value="updates">
              <TabContent type="updates" data={results.updates} />
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
};

export default AnalysisResults;
