
import { AnalysisResults } from "./types";

export const generateDummyResults = (url: string): AnalysisResults => {
  return {
    campaignTitle: "Smart Home Automation Kit",
    campaignUrl: url,
    timestamp: new Date().toISOString(),
    description: {
      score: 78,
      prediction: "success",
      confidence: 0.82,
      justification: "The campaign description contains clear product specifications, compelling use cases, and appropriate technical details. The tone is professional and enthusiastic.",
      findings: [
        "Strong product positioning",
        "Clear value proposition",
        "Well-defined target audience",
        "Good balance of technical specs and benefits",
        "Missing some manufacturing details"
      ]
    },
    comments: {
      score: 65,
      prediction: "success",
      confidence: 0.71,
      justification: "Comment sentiment is generally positive with some concerns about shipping timelines. Community engagement is good with prompt responses from the creator.",
      findings: [
        "Positive sentiment ratio: 70%",
        "Multiple comments expressing excitement",
        "Some concerns about delivery timeline",
        "Creator response rate: 85%",
        "Average response time: 6 hours"
      ]
    },
    youtube: {
      score: 82,
      prediction: "success",
      confidence: 0.88,
      justification: "Campaign video has good engagement metrics with high retention rate and positive sentiment in comments. Production quality is professional.",
      findings: [
        "View-to-pledge conversion rate: 2.3%",
        "Average view duration: 70% of total length",
        "Like-to-dislike ratio: 95:5",
        "Professional narration and demonstrations",
        "Clear product tutorials"
      ]
    },
    updates: {
      score: 59,
      prediction: "uncertain",
      confidence: 0.63,
      justification: "Update frequency is below average, but quality of updates is good. More frequent communication would improve backer confidence.",
      findings: [
        "Update frequency: Every 12 days (below average)",
        "Content quality is informative",
        "Response to backer questions in updates is timely",
        "Transparent about minor manufacturing delays",
        "Missing technical progress milestones"
      ]
    },
    overallPrediction: "success"
  };
};
