"""
Email datasets with ground truth for the three triage tasks.

Task 1 (email-classify)  — Easy   — 5 emails, classify by category
Task 2 (email-prioritize)— Medium — 8 emails, rank by urgency
Task 3 (email-triage-full)— Hard  — 12 emails, full triage + responses
"""

from typing import Any, Dict, List

# ─────────────────────────────────────────────────────────────
# TASK 1: EMAIL CLASSIFICATION  (Easy)
# ─────────────────────────────────────────────────────────────
TASK1_EMAILS: List[Dict[str, str]] = [
    {
        "id": "e1",
        "sender": "prize@mega-lottery-winner.net",
        "subject": "CONGRATULATIONS! You have won $1,000,000!",
        "body": (
            "Dear Lucky Winner, You have been selected as our grand prize winner! "
            "Claim your $1,000,000 prize immediately by clicking the link below. "
            "This offer expires in 24 hours! Don't miss this life-changing opportunity! "
            "Provide your bank details to transfer the funds directly."
        ),
        "timestamp": "2024-01-15T09:00:00Z",
    },
    {
        "id": "e2",
        "sender": "director@acme-corp.com",
        "subject": "Q3 Budget Review - Action Required by Friday",
        "body": (
            "Hi, I've attached the Q3 budget report for your review. "
            "Please provide your comments and approval by Friday EOD as we need "
            "this finalized before the board meeting on Monday. "
            "Let me know if you have any questions. Thanks, Sarah"
        ),
        "timestamp": "2024-01-15T10:30:00Z",
    },
    {
        "id": "e3",
        "sender": "mike.chen@gmail.com",
        "subject": "Saturday hiking trip — are you in?",
        "body": (
            "Hey! A few of us are planning a hiking trip to Blue Ridge this Saturday. "
            "We'll leave around 7am and should be back by evening. "
            "Let me know if you want to join! Should be a great day outside."
        ),
        "timestamp": "2024-01-15T11:00:00Z",
    },
    {
        "id": "e4",
        "sender": "updates@productdigest.io",
        "subject": "Product Digest Weekly — Issue #156",
        "body": (
            "This week's top stories: 1) AI tools reshaping the workplace "
            "2) Top 10 productivity apps of 2024 3) How to build better habits at work. "
            "Read full articles at productdigest.io. Unsubscribe at any time."
        ),
        "timestamp": "2024-01-15T08:00:00Z",
    },
    {
        "id": "e5",
        "sender": "oncall@infrastructure.company.com",
        "subject": "URGENT: Production database DOWN — all services affected",
        "body": (
            "CRITICAL ALERT: The production database has been unreachable for 45 minutes. "
            "ALL customer-facing services are down. We have escalated to DBAs but need "
            "immediate authorization to initiate failover. Please respond IMMEDIATELY. "
            "Every minute of downtime costs approximately $5,000."
        ),
        "timestamp": "2024-01-15T11:45:00Z",
    },
]

TASK1_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "e1": {"category": "spam",       "priority": 10, "action": "delete"},
    "e2": {"category": "work",       "priority": 2,  "action": "respond"},
    "e3": {"category": "personal",   "priority": 6,  "action": "respond"},
    "e4": {"category": "newsletter", "priority": 8,  "action": "archive"},
    "e5": {"category": "urgent",     "priority": 1,  "action": "respond"},
}

# ─────────────────────────────────────────────────────────────
# TASK 2: EMAIL PRIORITIZATION  (Medium)
# ─────────────────────────────────────────────────────────────
TASK2_EMAILS: List[Dict[str, str]] = [
    {
        "id": "p1",
        "sender": "cto@company.com",
        "subject": "CRITICAL: Main API down — 2,000 users affected",
        "body": (
            "Our main API has been returning 503 errors for the last 2 hours. "
            "Support is overwhelmed with tickets. We've lost 3 enterprise clients already. "
            "Need a decision on rollback NOW. Call me immediately: 555-0100."
        ),
        "timestamp": "2024-01-15T09:00:00Z",
    },
    {
        "id": "p2",
        "sender": "client@bigcustomer.com",
        "subject": "Meeting starts in 30 minutes — dial-in info?",
        "body": (
            "Hi, I'm ready for our product demo but don't have the dial-in info. "
            "Our CEO and CTO are in the conference room with me. "
            "Please send the link ASAP! Looking forward to it."
        ),
        "timestamp": "2024-01-15T10:30:00Z",
    },
    {
        "id": "p3",
        "sender": "legal@law-firm.com",
        "subject": "Contract must be signed by 5pm TODAY or deal is void",
        "body": (
            "As per our last conversation, the acquisition contract expires at 5pm today. "
            "The other party will not extend the deadline. "
            "Please DocuSign the attached contract before 5pm or the $2M deal falls through."
        ),
        "timestamp": "2024-01-15T09:30:00Z",
    },
    {
        "id": "p4",
        "sender": "hr@company.com",
        "subject": "Performance review forms due next Wednesday",
        "body": (
            "Reminder: Annual performance review forms are due by Wednesday, January 24th. "
            "Please complete the self-assessment for all direct reports. "
            "Forms can be submitted through the HR portal."
        ),
        "timestamp": "2024-01-15T08:00:00Z",
    },
    {
        "id": "p5",
        "sender": "finance@company.com",
        "subject": "Q4 budget approval needed before end of week",
        "body": (
            "The Q4 marketing budget proposal is ready for your approval. "
            "We need sign-off by Friday so procurement can begin vendor negotiations. "
            "Proposed budget: $450,000 — a 15% increase from Q3. See attached breakdown."
        ),
        "timestamp": "2024-01-15T10:00:00Z",
    },
    {
        "id": "p6",
        "sender": "colleague@company.com",
        "subject": "Coffee chat this week?",
        "body": (
            "Hey! I've been meaning to catch up. Would you be free for a quick coffee "
            "sometime this week? No agenda, just a casual chat. Let me know what works!"
        ),
        "timestamp": "2024-01-15T11:00:00Z",
    },
    {
        "id": "p7",
        "sender": "newsletter@startup-weekly.com",
        "subject": "Startup Weekly: 10 Lessons from Failed Startups",
        "body": (
            "This week's feature: What we learned from 50 failed startups. "
            "Plus: Fundraising in 2024, top VCs to watch, and upcoming events. "
            "Forward to a colleague | Unsubscribe"
        ),
        "timestamp": "2024-01-15T07:00:00Z",
    },
    {
        "id": "p8",
        "sender": "deals@shoppingdeals.com",
        "subject": "FLASH SALE: 70% off everything, today only!",
        "body": (
            "Don't miss our biggest sale of the year! Everything 70% off "
            "for the next 4 hours! Use code FLASH70 at checkout. "
            "Shop now before it's too late! Terms and conditions apply."
        ),
        "timestamp": "2024-01-15T06:00:00Z",
    },
]

# Priority 1 = most urgent/important; 8 = least
TASK2_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "p1": {"priority": 1, "category": "urgent",     "action": "respond"},  # Critical infra down
    "p2": {"priority": 2, "category": "urgent",     "action": "respond"},  # Client waiting NOW
    "p3": {"priority": 3, "category": "work",       "action": "respond"},  # Contract expires today
    "p4": {"priority": 4, "category": "work",       "action": "read"},     # Due next week
    "p5": {"priority": 5, "category": "work",       "action": "respond"},  # Due this week
    "p6": {"priority": 6, "category": "personal",   "action": "respond"},  # Casual, no urgency
    "p7": {"priority": 7, "category": "newsletter", "action": "archive"},  # Newsletter
    "p8": {"priority": 8, "category": "spam",       "action": "delete"},   # Marketing spam
}

# ─────────────────────────────────────────────────────────────
# TASK 3: FULL EMAIL TRIAGE  (Hard)
# ─────────────────────────────────────────────────────────────
TASK3_EMAILS: List[Dict[str, str]] = [
    {
        "id": "f1",
        "sender": "security@mybank.com",
        "subject": "Suspicious login detected on your account",
        "body": (
            "We detected a login attempt from an unrecognized device in Moscow, Russia "
            "at 3:47 AM. If this was you, no action needed. "
            "If not, please secure your account and change your password immediately."
        ),
        "timestamp": "2024-01-15T09:00:00Z",
    },
    {
        "id": "f2",
        "sender": "ceo@partner-company.com",
        "subject": "Partnership proposal — call this week?",
        "body": (
            "Hi, I'm the CEO of TechVentures. We're interested in a strategic partnership "
            "as our product complements yours perfectly. Could we schedule a 30-minute call "
            "this week to explore mutual value? Happy to work around your schedule."
        ),
        "timestamp": "2024-01-15T10:00:00Z",
    },
    {
        "id": "f3",
        "sender": "prize@free-gift-winner.biz",
        "subject": "You have been selected for a FREE iPhone 15!",
        "body": (
            "Congratulations! You were randomly selected to receive a FREE iPhone 15 Pro! "
            "To claim, provide your shipping address and credit card for a $1.99 shipping fee. "
            "Offer expires in 2 hours! Click here to claim!"
        ),
        "timestamp": "2024-01-15T08:00:00Z",
    },
    {
        "id": "f4",
        "sender": "team-lead@company.com",
        "subject": "URGENT: Client demo in 2 hours — presentation missing slides",
        "body": (
            "The client demo is at 2pm today and slides 15-20 covering our roadmap are missing. "
            "These are critical for the demo — our biggest prospect will be there. "
            "Can you please send the roadmap slides ASAP?"
        ),
        "timestamp": "2024-01-15T11:00:00Z",
    },
    {
        "id": "f5",
        "sender": "devops@company.com",
        "subject": "CRITICAL: Payment processing service down — revenue impact",
        "body": (
            "ALERT: Payment processing service has been down for 35 minutes. "
            "We're losing approximately $8,000 per minute. 847 failed transactions so far. "
            "Engineers need authorization to deploy the emergency patch. PLEASE RESPOND NOW."
        ),
        "timestamp": "2024-01-15T11:30:00Z",
    },
    {
        "id": "f6",
        "sender": "mom@family.com",
        "subject": "Thanksgiving dinner plans",
        "body": (
            "Hi honey! Starting to plan Thanksgiving dinner. Will you be able to make it? "
            "Aunt Ruth and Uncle Bob are coming, and Jennifer is bringing her new boyfriend. "
            "Let me know so I can finalize the headcount!"
        ),
        "timestamp": "2024-01-15T09:30:00Z",
    },
    {
        "id": "f7",
        "sender": "vendor@software-vendor.com",
        "subject": "Your license renewal is due in 30 days",
        "body": (
            "Your enterprise license for DataAnalytics Pro expires on February 15, 2024. "
            "To avoid service interruption, please renew before expiration. "
            "Contact your account manager at sales@software-vendor.com."
        ),
        "timestamp": "2024-01-15T08:30:00Z",
    },
    {
        "id": "f8",
        "sender": "digest@aibriefing.com",
        "subject": "AI Briefing: GPT-5 rumors, Gemini updates, Claude news",
        "body": (
            "Today's AI briefing: 1) OpenAI hints at GPT-5 timeline "
            "2) Google Gemini Ultra benchmark results 3) Claude 3 capabilities "
            "4) EU AI regulation updates. Click to read full summaries."
        ),
        "timestamp": "2024-01-15T07:00:00Z",
    },
    {
        "id": "f9",
        "sender": "customer@angry-client.com",
        "subject": "Completely unacceptable service — demand immediate response",
        "body": (
            "Your product has been broken for 3 days and nobody is helping me! "
            "I've sent 5 support tickets and called 3 times with no resolution. "
            "I'm a premium customer paying $5,000/month — this is UNACCEPTABLE. "
            "If not resolved TODAY I will cancel and post negative reviews everywhere."
        ),
        "timestamp": "2024-01-15T10:30:00Z",
    },
    {
        "id": "f10",
        "sender": "hr@company.com",
        "subject": "Team building event next Friday — RSVP needed",
        "body": (
            "Excited to announce our quarterly team building event next Friday, Jan 22nd! "
            "Escape room followed by dinner. Please RSVP by Wednesday "
            "so we can finalize headcount with the venue. Hope to see everyone there!"
        ),
        "timestamp": "2024-01-15T09:00:00Z",
    },
    {
        "id": "f11",
        "sender": "recruiter@tech-recruit.com",
        "subject": "Exciting opportunity at Google — $300k package",
        "body": (
            "Hi, I came across your profile and think you'd be perfect for a Senior "
            "Engineering role at Google. Package is $300k+ total comp. "
            "Open to a quick 15-minute call? No commitment. Reply STOP to unsubscribe."
        ),
        "timestamp": "2024-01-15T08:30:00Z",
    },
    {
        "id": "f12",
        "sender": "newsletter@producthunt.com",
        "subject": "Product Hunt Daily: Top Products of the Day",
        "body": (
            "Today's top products: 1) AI Writing Assistant 2) Smart Scheduling App "
            "3) Developer Tools Platform 4) Analytics Dashboard. "
            "Vote for your favorites and discover tomorrow's trending products!"
        ),
        "timestamp": "2024-01-15T06:00:00Z",
    },
]

TASK3_GROUND_TRUTH: Dict[str, Dict[str, Any]] = {
    "f1": {
        "category": "urgent",     "priority": 2,  "action": "respond",
        "needs_response": True,
        "response_keywords": ["security", "password", "account", "investigate", "verify"],
    },
    "f2": {
        "category": "work",       "priority": 4,  "action": "respond",
        "needs_response": True,
        "response_keywords": ["partnership", "call", "schedule", "interested", "discuss"],
    },
    "f3": {
        "category": "spam",       "priority": 12, "action": "delete",
        "needs_response": False,  "response_keywords": [],
    },
    "f4": {
        "category": "urgent",     "priority": 3,  "action": "respond",
        "needs_response": True,
        "response_keywords": ["slides", "demo", "sending", "roadmap", "presentation"],
    },
    "f5": {
        "category": "urgent",     "priority": 1,  "action": "respond",
        "needs_response": True,
        "response_keywords": ["payment", "authorize", "patch", "approve", "emergency"],
    },
    "f6": {
        "category": "personal",   "priority": 8,  "action": "respond",
        "needs_response": True,
        "response_keywords": ["thanksgiving", "dinner", "yes", "plan", "attend"],
    },
    "f7": {
        "category": "work",       "priority": 6,  "action": "flag",
        "needs_response": False,  "response_keywords": [],
    },
    "f8": {
        "category": "newsletter", "priority": 10, "action": "archive",
        "needs_response": False,  "response_keywords": [],
    },
    "f9": {
        "category": "urgent",     "priority": 3,  "action": "respond",
        "needs_response": True,
        "response_keywords": ["apologize", "sorry", "resolve", "escalate", "support"],
    },
    "f10": {
        "category": "work",       "priority": 7,  "action": "respond",
        "needs_response": True,
        "response_keywords": ["attend", "rsvp", "friday", "looking forward", "join"],
    },
    "f11": {
        "category": "newsletter", "priority": 9,  "action": "archive",
        "needs_response": False,  "response_keywords": [],
    },
    "f12": {
        "category": "newsletter", "priority": 11, "action": "archive",
        "needs_response": False,  "response_keywords": [],
    },
}

# ─────────────────────────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────────────────────────
TASKS: Dict[str, Dict[str, Any]] = {
    "email-classify": {
        "id": "email-classify",
        "name": "Email Classification",
        "difficulty": "easy",
        "description": (
            "Classify 5 emails into the correct categories: "
            "spam, work, personal, newsletter, or urgent."
        ),
        "instructions": (
            "You will receive 5 emails. For each email, determine:\n"
            "1. category: spam | work | personal | newsletter | urgent\n"
            "2. priority: integer 1 (most important) to 10 (least important)\n"
            "3. action: read | archive | delete | respond | flag\n"
            "Submit your decisions as a JSON object with an 'email_actions' array."
        ),
        "emails": TASK1_EMAILS,
        "ground_truth": TASK1_GROUND_TRUTH,
        "max_steps": 5,
        "weights": {"classification": 0.6, "priority": 0.2, "action": 0.2, "response": 0.0},
    },
    "email-prioritize": {
        "id": "email-prioritize",
        "name": "Email Prioritization",
        "difficulty": "medium",
        "description": (
            "Prioritize 8 emails by urgency. "
            "Assign rank 1 (most urgent) to 8 (least urgent) — no ties."
        ),
        "instructions": (
            "You will receive 8 emails. For each email:\n"
            "1. priority: unique integer 1 (most urgent) to 8 (least urgent) — no two emails share the same priority\n"
            "2. category: spam | work | personal | newsletter | urgent\n"
            "3. action: read | archive | delete | respond | flag\n"
            "Consider business impact, time sensitivity, and stakeholder importance.\n"
            "Submit your decisions as a JSON object with an 'email_actions' array."
        ),
        "emails": TASK2_EMAILS,
        "ground_truth": TASK2_GROUND_TRUTH,
        "max_steps": 8,
        "weights": {"classification": 0.25, "priority": 0.6, "action": 0.15, "response": 0.0},
    },
    "email-triage-full": {
        "id": "email-triage-full",
        "name": "Full Email Triage",
        "difficulty": "hard",
        "description": (
            "Complete triage of 12 emails: classify, prioritize, choose actions, "
            "and draft responses for emails that need replies."
        ),
        "instructions": (
            "You will receive 12 emails requiring complete triage. For each email:\n"
            "1. category: spam | work | personal | newsletter | urgent\n"
            "2. priority: unique integer 1 (most urgent) to 12 (least urgent)\n"
            "3. action: read | archive | delete | respond | flag\n"
            "4. response_draft (REQUIRED for urgent/work emails that need replies):\n"
            "   - At least 20 words, professional tone\n"
            "   - Must address the specific content of the email\n"
            "   - Include a greeting and sign-off\n"
            "Your score is a weighted combination of classification accuracy, "
            "priority ranking quality, correct actions, and response quality.\n"
            "Submit your decisions as a JSON object with an 'email_actions' array."
        ),
        "emails": TASK3_EMAILS,
        "ground_truth": TASK3_GROUND_TRUTH,
        "max_steps": 12,
        "weights": {"classification": 0.30, "priority": 0.30, "action": 0.20, "response": 0.20},
    },
}
