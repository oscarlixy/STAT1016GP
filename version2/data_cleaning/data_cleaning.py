"""
================================================================================
AI Job Market Data Cleaning - Version 2
================================================================================
Three-layer AI skill classification system:
1. Layer 1: Regex-based coarse classification (job title keywords)
2. Layer 2: Extract gray zone data (uncertain cases)
3. Layer 3: API batch processing (Gemini LLM for nuanced classification)

Author: Oscar
Date: 2026-03-19
================================================================================
"""

import pandas as pd
import numpy as np
import re
import json
import os
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

# Configuration
BASE_DIR = '/Users/oscar/Desktop/STAT1016GP'
DATA_DIR = os.path.join(BASE_DIR, 'data')
VERSION2_DIR = os.path.join(BASE_DIR, 'version2')
OUTPUT_DIR = os.path.join(VERSION2_DIR, 'data')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# LAYER 1: REGEX-BASED COARSE CLASSIFICATION
# ============================================================================

@dataclass
class AITagResult:
    """Result of AI tag classification"""
    tag: int  # 0 or 1
    reason: str
    confidence: str  # 'high', 'medium', 'low'
    layer: int  # 1 (regex), 2 (gray zone), 3 (API)


class RegexClassifier:
    """
    Layer 1: Regex-based coarse classification using job title keywords.
    """

    # Keywords that strongly indicate AI core skills (ai_tag = 1)
    AI_CORE_KEYWORDS = [
        # Direct AI/ML terms
        r'\bAI\b', r'\bML\b', r'\bMLOps\b', r'\bMLOps\b',
        r'\bMachine Learning\b', r'\bDeep Learning\b',
        r'\bNeural Network\b', r'\bNLP\b', r'\bNatural Language Processing\b',
        r'\bComputer Vision\b', r'\bCV\b',
        r'\bArtificial Intelligence\b',
        r'\bMachine Learning Engineer\b', r'\bML Engineer\b',
        r'\bData Scientist\b',

        # AI Research roles
        r'\bAI Research\b', r'\bML Research\b', r'\bResearch Scientist\b',
        r'\bResearch Engineer\b',

        # AI Infrastructure
        r'\bML Infrastructure\b', r'\bAI Infrastructure\b',
        r'\bML Platform\b', r'\bAI Platform\b',

        # AI Application roles
        r'\bAI Software\b', r'\bAI Engineer\b',
        r'\bGenerative AI\b', r'\bLLM\b', r'\bLarge Language Model\b',
        r'\bFoundation Model\b',

        # Autonomous/AI Systems
        r'\bAutonomous\b', r'\bSelf-Driving\b', r'\bSelf-driving\b',
        r'\bRobotics\b', r'\bRobot\b',

        # Specialized AI roles
        r'\bReinforcement Learning\b', r'\bRL\b',
        r'\bTransformer\b', r'\bAttention\b',
        r'\bGAN\b', r'\bDiffusion\b',
        r'\bCV Engineer\b', r'\bVision Engineer\b',
        r'\bSpeech\b', r'\bVoice\b', r'\bASR\b',

        # AI Product/Management
        r'\bAI Product\b', r'\bAI Manager\b',
    ]

    # Keywords that indicate non-AI core roles (ai_tag = 0)
    NON_AI_KEYWORDS = [
        # Traditional Software Engineering
        r'\bSoftware Engineer\b', r'\bBackend\b', r'\bFrontend\b',
        r'\bFull Stack\b', r'\bFull-Stack\b', r'\bFullstack\b',
        r'\bWeb Developer\b', r'\bApp Developer\b',
        r'\biOS Developer\b', r'\bAndroid Developer\b',
        r'\bMobile Developer\b', r'\bReact Developer\b',
        r'\bPython Developer\b', r'\bJava Developer\b',
        r'\b.NET Developer\b', r'\bC# Developer\b',

        # DevOps/SRE
        r'\bDevOps\b', r'\bSRE\b', r'\bSite Reliability\b',
        r'\bPlatform Engineer\b', r'\bCloud Engineer\b',

        # Data Engineering (traditional)
        r'\bData Engineer\b', r'\bETL\b',
        r'\bData Warehouse\b', r'\bData Platform\b',

        # Business/Data Analysis
        r'\bData Analyst\b', r'\bBusiness Analyst\b',
        r'\bBI Analyst\b', r'\bAnalytics Engineer\b',
        r'\bBusiness Intelligence\b', r'\bBI Engineer\b',

        # Infrastructure/Operations
        r'\bSystem Administrator\b', r'\bSysAdmin\b',
        r'\bNetwork Engineer\b', r'\bSecurity Engineer\b',
        r'\bDatabase Administrator\b', r'\bDBA\b',
        r'\bInfrastructure Engineer\b',

        # Product/Management (non-AI)
        r'\bProduct Manager\b', r'\bProject Manager\b',
        r'\bScrum Master\b', r'\bAgile Coach\b',

        # QA/Testing
        r'\bQA Engineer\b', r'\bTest Engineer\b',
        r'\bQuality Assurance\b', r'\bSDET\b',

        # Support/Sales
        r'\bSupport Engineer\b', r'\bSales Engineer\b',
        r'\bCustomer Success\b', r'\bSolutions Architect\b',

        # Traditional IT
        r'\bIT Specialist\b', r'\bIT Analyst\b',
        r'\bTechnical Writer\b', r'\bUX Designer\b',
    ]

    # Gray zone keywords (uncertain, needs LLM)
    GRAY_ZONE_KEYWORDS = [
        r'\bAI Specialist\b', r'\bAI Architect\b',
        r'\bAI Consultant\b', r'\bAI Lead\b',
        r'\bAI Director\b', r'\bAI Manager\b',
        r'\bAI Researcher\b', r'\bAI Analyst\b',
        r'\bAI Coach\b', r'\bAI Trainer\b',
        r'\bAI Evangelist\b', r'\bAI Strategist\b',
        r'\bAI Advisor\b', r'\bAI Scientist\b',
        r'\bData Science\b', r'\bData Science Manager\b',
        r'\bPrincipal Data Scientist\b', r'\bSenior Data Scientist\b',
        r'\bStaff Data Scientist\b', r'\bLead Data Scientist\b',
    ]

    def __init__(self):
        """Initialize regex patterns."""
        self.ai_pattern = re.compile(
            '|'.join(self.AI_CORE_KEYWORDS),
            re.IGNORECASE
        )
        self.non_ai_pattern = re.compile(
            '|'.join(self.NON_AI_KEYWORDS),
            re.IGNORECASE
        )
        self.gray_zone_pattern = re.compile(
            '|'.join(self.GRAY_ZONE_KEYWORDS),
            re.IGNORECASE
        )

    def classify(self, job_title: str) -> Tuple[int, str, str]:
        """
        Classify job title using regex.

        Returns:
            Tuple of (ai_tag, reason, confidence)
        """
        if not isinstance(job_title, str):
            return (0, "Non-AI job title", "high")

        # Check gray zone first (higher priority for uncertain cases)
        if self.gray_zone_pattern.search(job_title):
            return (None, "Gray zone - needs LLM", "low")

        # Check for AI core keywords
        if self.ai_pattern.search(job_title):
            # Additional check: if it also matches non-AI, it's ambiguous
            if self.non_ai_pattern.search(job_title):
                return (None, "Ambiguous title - needs LLM", "low")
            return (1, "AI core keyword match", "high")

        # Check for non-AI keywords
        if self.non_ai_pattern.search(job_title):
            return (0, "Non-AI job title", "high")

        # No match - treat as non-AI (conservative approach)
        return (0, "Default non-AI", "medium")


# ============================================================================
# LAYER 2: GRAY ZONE EXTRACTION
# ============================================================================

def extract_gray_zone(df: pd.DataFrame, classifier: RegexClassifier) -> pd.DataFrame:
    """
    Extract gray zone data where regex classification is uncertain.

    Args:
        df: Input DataFrame
        classifier: RegexClassifier instance

    Returns:
        DataFrame containing only gray zone rows
    """
    gray_zone_indices = []

    for idx, row in df.iterrows():
        job_title = row.get('job_title', '')
        tag, reason, confidence = classifier.classify(job_title)

        # Include if: tag is None (uncertain), or confidence is low
        if tag is None or confidence == 'low':
            gray_zone_indices.append(idx)

    gray_zone_df = df.loc[gray_zone_indices].copy()
    gray_zone_df['_gray_zone_reason'] = [
        classifier.classify(row.get('job_title', ''))[1]
        for _, row in gray_zone_df.iterrows()
    ]

    print(f"\n[Layer 2] Gray Zone Extraction:")
    print(f"  Total records: {len(df)}")
    print(f"  Gray zone records: {len(gray_zone_df)}")
    print(f"  Gray zone percentage: {len(gray_zone_df)/len(df)*100:.1f}%")

    return gray_zone_df


# ============================================================================
# LAYER 3: API BATCH PROCESSING (Gemini)
# ============================================================================

class GeminiAPIClient:
    """
    Layer 3: Gemini API client for nuanced AI job classification.
    """

    # Prompt template for classification
    PROMPT_TEMPLATE = '''你是一位资深的人工智能与人力资源数据科学家，深谙现代科技行业的真实技术栈分布。你的任务是分析招聘数据，判定该岗位是否属于"核心 AI 技能相关岗位"。

【判定规则】
判定结果仅限 1 或 0：

输出 `1` (需要核心 AI 技能)：
岗位的核心职责涉及人工智能底层研发、模型训练、AI 基建 (MLOps) 或 AI 核心应用开发。
*特例注意*：部分高级 AI 岗位（如 NLP Engineer, ML Infrastructure Engineer, Computer Vision Researcher）可能在招聘要求中仅罗列底层语言或通用组件（如 Scala, C++, Python, Linux, CUDA），而不显式提及 PyTorch 等高层框架。只要岗位名称强烈指向 AI 核心领域，即判定为 1。

输出 `0` (不需要核心 AI 技能)：
岗位属于传统软件工程（前后端、移动端）、常规数据分析 (BI)、或普通运维开发。
*特例注意*：即使技能栈中包含了 Python、Docker、Kubernetes、AWS、SQL 等常用于 AI 但本质通用的工具，只要该岗位名称和核心业务不是构建或优化 AI 模型，一律严格判定为 0。

【待判定岗位信息】
职位名称: {job_title}
技能要求: {required_skills}

【输出格式】
必须且只能输出一个合法的 JSON 对象，绝对不要包含任何 Markdown 标记（如 ```json 标签）、换行符或其他多余文本。JSON 必须包含以下两个字段：
{{"ai_tag": <填入数字 1 或 0>, "reason": "<不超过20个字的极简判定理由>"}}
'''

    def __init__(self, api_key: str, max_retries: int = 5, base_delay: float = 5.0):
        """
        Initialize Gemini API client.

        Args:
            api_key: Gemini API key
            max_retries: Maximum number of retries for rate limit errors
            base_delay: Base delay in seconds for exponential backoff
        """
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.max_retries = max_retries
        self.base_delay = base_delay

    def classify_single(self, job_title: str, required_skills: str) -> Optional[Dict]:
        """
        Classify a single job posting using Gemini API with retry logic.

        Args:
            job_title: Job title
            required_skills: Required skills string

        Returns:
            Dict with 'ai_tag' and 'reason', or None if failed
        """
        import urllib.request
        import urllib.error
        import time

        prompt = self.PROMPT_TEMPLATE.format(
            job_title=job_title,
            required_skills=required_skills if isinstance(required_skills, str) else ""
        )

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 100,
                "topP": 0.8,
                "topK": 40
            }
        }

        url = f"{self.base_url}?key={self.api_key}"

        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode('utf-8'),
                    headers={'Content-Type': 'application/json'},
                    method='POST'
                )

                with urllib.request.urlopen(req, timeout=60) as response:
                    result = json.loads(response.read().decode('utf-8'))

                    # Extract text from response
                    text = result['candidates'][0]['content']['parts'][0]['text']

                    # Parse JSON from response
                    # Clean the response - remove any markdown formatting
                    text = text.strip()
                    if text.startswith('```'):
                        text = text.split('\n', 1)[1]
                    if text.endswith('```'):
                        text = text.rsplit('\n', 1)[0]
                    text = text.strip()

                    return json.loads(text)

            except urllib.error.HTTPError as e:
                if e.code == 429:  # Rate limit
                    delay = self.base_delay * (2 ** attempt)
                    print(f"  Rate limited, waiting {delay:.1f}s (attempt {attempt+1}/{self.max_retries})...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"API HTTP Error for '{job_title}': {e}")
                    return None

            except (urllib.error.URLError, json.JSONDecodeError, KeyError, Exception) as e:
                if attempt < self.max_retries - 1:
                    delay = self.base_delay * (2 ** attempt)
                    print(f"API Error for '{job_title}': {e}, retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                print(f"API Error for '{job_title}': {e}")
                return None

        return None

    def batch_classify(self, df: pd.DataFrame, rate_limit: float = 5.0) -> List[Optional[Dict]]:
        """
        Batch classify jobs with rate limiting.

        Args:
            df: DataFrame with job_title and required_skills columns
            rate_limit: Minimum seconds between API calls

        Returns:
            List of classification results
        """
        import time

        results = []
        total = len(df)
        success_count = 0
        fail_count = 0

        print(f"\n[Layer 3] API Batch Processing:")
        print(f"  Total gray zone records: {total}")
        print(f"  Rate limit: {rate_limit}s between calls")
        print(f"  Max retries per request: {self.max_retries}")

        for i, (_, row) in enumerate(df.iterrows()):
            job_title = row.get('job_title', '')
            required_skills = row.get('required_skills', '')

            print(f"  [{i+1}/{total}] Processing: {job_title[:50]}...", end=' ')

            result = self.classify_single(job_title, required_skills)

            if result:
                print(f"-> ai_tag={result.get('ai_tag')}")
                success_count += 1
            else:
                print("-> FAILED (will use default)")
                fail_count += 1

            results.append(result)

            # Rate limiting (only if successful or final failure)
            if i < total - 1:
                time.sleep(rate_limit)

        print(f"\n  API Processing Summary:")
        print(f"    Success: {success_count}")
        print(f"    Failed: {fail_count}")
        print(f"    Total: {total}")

        return results


# ============================================================================
# MAIN DATA CLEANING PIPELINE
# ============================================================================

def run_data_cleaning_pipeline(
    api_key: Optional[str] = None,
    use_api: bool = False,
    rate_limit: float = 5.0
) -> pd.DataFrame:
    """
    Run the complete data cleaning pipeline.

    Args:
        api_key: Gemini API key (required if use_api=True)
        use_api: Whether to use API for gray zone classification
        rate_limit: Seconds between API calls

    Returns:
        Cleaned DataFrame with AI_Skill_Requirement column
    """
    print("="*70)
    print("AI JOB MARKET DATA CLEANING PIPELINE - VERSION 2")
    print("="*70)
    print("\nThree-layer AI skill classification system:")
    print("  Layer 1: Regex-based coarse classification")
    print("  Layer 2: Extract gray zone data")
    print("  Layer 3: API batch processing (Gemini)")
    print("="*70)

    # Load raw data
    raw_data_path = os.path.join(DATA_DIR, 'ai_job_dataset.csv')
    print(f"\n[Loading Data]")
    print(f"  Path: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    print(f"  Total records: {len(df)}")

    # ============================================================
    # LAYER 1: REGEX CLASSIFICATION
    # ============================================================
    print("\n" + "-"*70)
    print("[Layer 1] Regex-Based Coarse Classification")
    print("-"*70)

    classifier = RegexClassifier()

    layer1_tags = []
    layer1_reasons = []
    layer1_confidence = []

    for idx, row in df.iterrows():
        job_title = row.get('job_title', '')
        tag, reason, confidence = classifier.classify(job_title)
        layer1_tags.append(tag)
        layer1_reasons.append(reason)
        layer1_confidence.append(confidence)

    df['_layer1_tag'] = layer1_tags
    df['_layer1_reason'] = layer1_reasons
    df['_layer1_confidence'] = layer1_confidence

    # Summary statistics
    certain_count = sum(1 for t in layer1_tags if t is not None)
    uncertain_count = sum(1 for t in layer1_tags if t is None)

    print(f"\nLayer 1 Results:")
    print(f"  Certain classifications: {certain_count} ({certain_count/len(df)*100:.1f}%)")
    print(f"  Uncertain (gray zone): {uncertain_count} ({uncertain_count/len(df)*100:.1f}%)")

    # ============================================================
    # LAYER 2: GRAY ZONE EXTRACTION
    # ============================================================
    print("\n" + "-"*70)
    print("[Layer 2] Gray Zone Extraction")
    print("-"*70)

    gray_zone_df = extract_gray_zone(df, classifier)

    # Save gray zone data for manual inspection/API processing
    gray_zone_path = os.path.join(OUTPUT_DIR, 'gray_zone_data.csv')
    gray_zone_df.to_csv(gray_zone_path, index=False)
    print(f"  Gray zone data saved to: {gray_zone_path}")

    # ============================================================
    # LAYER 3: API BATCH PROCESSING
    # ============================================================
    if use_api and api_key:
        print("\n" + "-"*70)
        print("[Layer 3] API Batch Processing (Gemini)")
        print("-"*70)

        # Initialize API client with retry settings
        api_client = GeminiAPIClient(
            api_key,
            max_retries=5,
            base_delay=10.0  # Start with 10s, exponential backoff to 160s max
        )
        api_results = api_client.batch_classify(gray_zone_df, rate_limit=rate_limit)

        # Update gray zone DataFrame with API results
        api_tags = []
        api_reasons = []

        for result in api_results:
            if result and 'ai_tag' in result:
                api_tags.append(result['ai_tag'])
                api_reasons.append(result.get('reason', ''))
            else:
                # Default to 0 if API fails
                api_tags.append(0)
                api_reasons.append('API failed - default to 0')

        gray_zone_df['_layer3_api_tag'] = api_tags
        gray_zone_df['_layer3_api_reason'] = api_reasons

        # Save updated gray zone data
        gray_zone_df.to_csv(gray_zone_path, index=False)
        print(f"  Updated gray zone data saved to: {gray_zone_path}")

        # Update main DataFrame with API results
        gray_indices = gray_zone_df.index.tolist()
        for i, idx in enumerate(gray_indices):
            df.loc[idx, '_layer1_tag'] = api_tags[i]
            df.loc[idx, '_layer1_reason'] = api_reasons[i]
            df.loc[idx, '_layer1_confidence'] = 'api'

    # ============================================================
    # FINALIZE AI_SKILL_REQUIREMENT
    # ============================================================
    print("\n" + "-"*70)
    print("[Finalize] AI_Skill_Requirement Column")
    print("-"*70)

    # Fill None values with 0 (conservative approach for failed API calls)
    df['AI_Skill_Requirement'] = df['_layer1_tag'].fillna(0).astype(int)

    # ============================================================
    # DATA TRANSFORMATION
    # ============================================================
    print("\n" + "-"*70)
    print("[Data Transformation]")
    print("-"*70)

    # Log salary
    df['Log_Salary'] = np.log(df['salary_usd'])

    # Scale experience (0-15 years to 0-1)
    max_exp = df['years_experience'].max()
    df['Experience_Scaled'] = df['years_experience'] / max_exp

    # One-hot encode industry
    industry_dummies = pd.get_dummies(df['industry'], prefix='Industry')
    df = pd.concat([df, industry_dummies], axis=1)

    # ============================================================
    # SELECT FINAL COLUMNS
    # ============================================================
    final_columns = [
        'salary_usd', 'Log_Salary', 'years_experience', 'Experience_Scaled',
        'AI_Skill_Requirement'
    ] + [col for col in df.columns if col.startswith('Industry_')]

    df_cleaned = df[final_columns].copy()

    # ============================================================
    # SAVE OUTPUT
    # ============================================================
    output_path = os.path.join(OUTPUT_DIR, 'cleaned_ai_job_data_v2.csv')
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n[Output] Cleaned data saved to: {output_path}")

    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"\nAI_Skill_Requirement Distribution:")
    print(df_cleaned['AI_Skill_Requirement'].value_counts())
    print(f"\n  Total records: {len(df_cleaned)}")
    print(f"  AI Skill Required (1): {df_cleaned['AI_Skill_Requirement'].sum()} ({df_cleaned['AI_Skill_Requirement'].mean()*100:.1f}%)")
    print(f"  Non-AI (0): {(1-df_cleaned['AI_Skill_Requirement']).sum()} ({(1-df_cleaned['AI_Skill_Requirement'].mean())*100:.1f}%)")

    return df_cleaned


# ============================================================================
# STANDALONE GRAY ZONE PROCESSING
# ============================================================================

def process_existing_gray_zone(api_key: str, rate_limit: float = 5.0):
    """
    Process existing gray zone data with API.

    Args:
        api_key: Gemini API key
        rate_limit: Seconds between API calls
    """
    gray_zone_path = os.path.join(OUTPUT_DIR, 'gray_zone_data.csv')

    if not os.path.exists(gray_zone_path):
        print("No gray zone data found. Run full pipeline first.")
        return

    print("Loading existing gray zone data...")
    gray_zone_df = pd.read_csv(gray_zone_path)

    api_client = GeminiAPIClient(
        api_key,
        max_retries=5,
        base_delay=10.0
    )
    api_results = api_client.batch_classify(gray_zone_df, rate_limit=rate_limit)

    # Update and save
    api_tags = []
    api_reasons = []

    for result in api_results:
        if result and 'ai_tag' in result:
            api_tags.append(result['ai_tag'])
            api_reasons.append(result.get('reason', ''))
        else:
            api_tags.append(0)
            api_reasons.append('API failed')

    gray_zone_df['_layer3_api_tag'] = api_tags
    gray_zone_df['_layer3_api_reason'] = api_reasons
    gray_zone_df.to_csv(gray_zone_path, index=False)

    print(f"\nUpdated {len(gray_zone_df)} gray zone records.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='AI Job Data Cleaning - Version 2')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--use-api', action='store_true', help='Use API for gray zone processing')
    parser.add_argument('--rate-limit', type=float, default=5.0, help='Seconds between API calls')
    parser.add_argument('--process-existing', action='store_true', help='Process existing gray zone data')

    args = parser.parse_args()

    if args.process_existing:
        if not args.api_key:
            print("Error: --api-key required for processing existing gray zone data")
            exit(1)
        process_existing_gray_zone(args.api_key, args.rate_limit)
    else:
        run_data_cleaning_pipeline(
            api_key=args.api_key,
            use_api=args.use_api,
            rate_limit=args.rate_limit
        )
