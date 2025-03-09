import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import logging
import json

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key= "sk-svcacct-iUaAf-5l-gYPDY9rpykP9kbYZLIFKTkl5mOXcVSNUQyv6w8EIMQBuFkY8W6p9p9m0T6TwJWT3BlbkFJ_BP2-NtINA4NVqI4cai5T5d47A80HEo63TgqHymWf0La0PYWXLkoHAIdrseD6K9NVMUWLAA")

# Retry logic for API calls
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_batch_descriptions(batch):
    """
    Generate descriptions for a batch of campaigns.
    """
    try:
        # Create a single prompt for the batch
        prompt = "Generate compelling campaign descriptions for the following Kickstarter projects:\n"
        for i, (name, subtitle) in enumerate(batch, start=1):
            prompt += f"\n{i}. Name: {name}\n   Subtitle: {subtitle}\n   Description:"

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in writing engaging Kickstarter campaign descriptions."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500  # Adjust based on the total length of descriptions
        )

        # Extract descriptions from the response
        descriptions = response.choices[0].message.content.strip().split("\n")
        descriptions = [desc.strip() for desc in descriptions if desc.strip()]

        # Ensure the number of descriptions matches the batch size
        if len(descriptions) != len(batch):
            logger.warning(f"Number of descriptions ({len(descriptions)}) does not match batch size ({len(batch)}).")
            descriptions.extend(["Description could not be generated"] * (len(batch) - len(descriptions)))

        return descriptions

    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise

# Load CSV file
csv_file = "kickstarter_successful_campaigns_test1.csv"
df = pd.read_csv(csv_file)

# Filter for campaigns under the "Technology" category
if "category" in df.columns:
    df = df[df["category"].str.lower() == "technology"]  # Case-insensitive filtering
    logger.info(f"Filtered dataset to {len(df)} rows under the 'Technology' category.")
else:
    logger.error("The 'category' column does not exist in the CSV file.")
    exit(1)

# Extract names and subtitles
if "name" not in df.columns or "subtitle" not in df.columns:
    logger.error("The CSV file must contain 'name' and 'subtitle' columns.")
    exit(1)

campaigns = list(zip(df["name"], df["subtitle"]))

# Generate descriptions in batches
batch_size = 10  # Process 5 campaigns at a time
results = []

for i in tqdm(range(0, len(campaigns), batch_size), desc="Processing batches"):
    batch = campaigns[i:i + batch_size]
    try:
        descriptions = generate_batch_descriptions(batch)
        for (name, subtitle), description in zip(batch, descriptions):
            results.append({
                "name": name,
                "subtitle": subtitle,
                "description": description
            })
    except Exception as e:
        logger.error(f"Failed to process batch starting at index {i}: {e}")
        for (name, subtitle) in batch:
            results.append({
                "name": name,
                "subtitle": subtitle,
                "description": "Description could not be generated"
            })

# Save results to JSON
json_output_file = "technology_campaign_descriptions.json"
with open(json_output_file, "w") as f:
    json.dump(results, f, indent=4)
logger.info(f"Descriptions saved to {json_output_file}")

# Merge descriptions back into the original DataFrame
df["description"] = [result["description"] for result in results]

# Save the updated CSV file
output_file = "technology_kickstarter_campaigns_with_descriptions.csv"
df.to_csv(output_file, index=False)
logger.info(f"Updated CSV saved to {output_file}")