---
title: Resume — David Inyang-Etoh
source: resume_pdf
breadcrumb: Resume
---

# David Inyang-Etoh — Resume

dinyangetoh@gmail.com • linkedin.com/in/david-inyang-etoh • dinyangetoh.com • Uyo, Nigeria

## Professional Summary

Senior Full-Stack Engineer, Team Lead, and AI Engineer with 8+ years of experience architecting high-impact SaaS and FinTech platforms. Specialist in TypeScript-first, cloud-native microservices and distributed systems, with a proven track record leading legacy-to-modern migration programmes. Expanding into AI Engineering — building production LangGraph multi-agent systems, RAG pipelines, and agentic AI products. Key achievements include reducing AWS costs by 66%, improving application performance by over 70%, and leading remote engineering squads to successful product launches.

## Technical Expertise

- **Backend:** Node.js, NestJS, Fastify, Express, TypeScript, PHP (Laravel)
- **Frontend:** React 18, Next.js 15, Vue.js, Nuxt.js, Tailwind CSS
- **Databases:** PostgreSQL (Aurora), Redis/Valkey, DynamoDB, MongoDB, MySQL
- **Cloud & Infrastructure:** AWS (Lambda, Fargate, ECS, API Gateway, S3, RDS Aurora, CDK, SNS/SQS, MSK Kafka), Docker, Terraform
- **AI & ML Engineering:** LangGraph, LangChain, LangSmith, OpenAI Agents SDK, RAG pipelines (hybrid BM25 + dense), Qdrant, Chroma, Claude API (langchain-anthropic), Prompt Engineering, QLoRA fine-tuning, MCP servers
- **Architecture & DevOps:** Microservices, CQRS, DDD, Event-Driven Systems, CI/CD (GitHub Actions, Jenkins), IaC (CDK, Terraform, CloudFormation)
- **Integrations:** REST, GraphQL, Webhooks, OAuth2, JWT, Third-Party APIs (Paystack, Stripe, Twilio, OpenAI, SendGrid)
- **Monitoring:** AWS CloudWatch, Sentry, Datadog, LangSmith tracing
- **Leadership:** Technical Team Lead, Agile/Scrum, Remote Team Management, TDD, Code Review, Migration Governance, Clean Architecture

## AI Engineering & Projects

### Vera AI / Veroliq | Founder & AI Engineer | 2024 – Present

- Founded Veroliq and architected Vera AI — an autonomous website conversion agent deployed as a script tag embed on customer sites, built with LangGraph and Claude via langchain-anthropic.
- Designed a multi-tenant RAG backend using Qdrant Cloud for hybrid vector search (dense + BM25 sparse retrieval), with configurable per-site knowledge bases.

### LaunchPad | Andela AI Bootcamp — Group Project | 2025

- Co-built a production-grade multi-agent LangGraph system targeting solo founders — three specialised agents: Concierge (24/7 support triage + auto-reply), Pulse (churn detection + retention nudges), and Intelligence (self-improving KB + quality scoring).
- Demonstrated 6 agentic concepts in one flow: orchestration, tool use, memory/state, agent handoff, human-in-loop (interrupt()), and cross-agent signalling — all observable via LangSmith traces.
- Integrated SendGrid, Slack Bot API, and Chroma/Qdrant; applied deliberate model-per-node strategy (gpt-4.1 for reasoning, gpt-4.1-mini for classification) at estimated $0.004–$0.008 per resolved ticket.

### Academic Multi-Agent Deep Researcher | Personal Project | 2025

- Built a LangGraph research pipeline with Gradio UI — human-in-loop clarifier, structured planner output, and parallel Send-based fan-out to Semantic Scholar, ArXiv + CORE, Tavily, and PubMed.
- Implemented SqliteSaver checkpointing with per-session thread_id, structured output at every stage (planner, rater, final report), confidence-based filtering, and optional LangSmith tracing + SendGrid email delivery.

### RAG Pipeline Optimisation | Andela AI Bootcamp — RAG Challenge | 2025

- Achieved MRR of 0.9508 and completeness score of 4.0 on the Insurellm dataset — top-performing configuration in cohort.
- Implemented hybrid retrieval (BM25 + dense) with text-embedding-3-large, ContextualCompressionRetriever, and cross-encoder reranking (RETRIEVAL_K=30, RERANK_TOP_N=8, lambda_mult=0.85).

## Professional Experience

### Team Lead / Lead Software Engineer (Contract) | Collectwire Technologies | Delaware, USA (Remote)
Nov 2024 – Present

- Led a squad of 4 engineers as technical team lead, architecting and delivering a global payroll SaaS MVP across 8 event-driven TypeScript microservices — setting code standards, running code reviews, and owning all architectural decisions through to beta launch within 7 months.
- Slashed AWS infrastructure costs by 66% (from $200+ to $67/month) by migrating to TypeScript CDK and strategically rightsizing cloud resources.
- Accelerated team delivery by 3x by embedding AI-assisted development tools (Cursor, Claude) into architecture design, documentation, and cost-benefit analysis workflows.
- Reduced redundant API calls by up to 80% across three frontend applications by standardising multi-tenant request headers and implementing a robust RTK Query caching strategy.

### Senior Software Engineer | Receeve GmbH | Hamburg, Germany (Remote)
Jan 2023 – Sep 2024

- Migrated a legacy monolithic orchestration service into three TypeScript serverless microservices on AWS Lambda, preserving full behavioral parity while improving execution speed by 47% and reducing cold start latency.
- Decreased API response times by 35% by implementing a CQRS architecture with a GraphQL Gateway, enabling independent scaling of read and write operations.
- Integrated OpenAI APIs to power AI-driven summaries and payment predictions, improving customer engagement by 10% and reducing manual review time by 25%.
- Achieved 99.9% deployment reliability via IaC with AWS CDK and GitHub Actions, including automated integration tests across multi-environment deployments.

### Software Engineer | Immutable X | Sydney, Australia (Remote)
Aug 2021 – Nov 2022

- Led revamp of the NFT checkout and payment flow, migrating legacy transaction logic into modern fintech payment integrations and improving wallet UX.
- Designed and implemented Wallet and Order microservices using NestJS, Kafka, and PostgreSQL, ensuring consistent and fault-tolerant transaction processing.
- Built Redis-based distributed locking (Redlock) to prevent duplicate NFT minting in concurrent transactions, improving reliability under high load.
- Automated containerised deployments on AWS ECS with CloudFormation; contributed to Solidity smart contract updates for secure on-chain asset transfers.

### Software Engineer | Jiggle Limited | Uyo, Nigeria
Nov 2019 – May 2021

- Co-founded the platform and architected a multi-currency digital wallet system with a distributed PostgreSQL ledger, supporting 10,000+ users with sub-second transaction processing.
- Increased successful payment transactions by 20% by integrating Paystack gateway with automated reconciliation and robust webhook handling.
- Scaled the engineering team from 3 to 7 members, establishing coding standards and mentoring junior developers on full-stack best practices.

### Full Stack Engineer | VisionDev | Provo, USA (Remote)
Jan 2017 – Oct 2019

- Reduced manual data entry by 28% and improved payroll accuracy for 500+ employees by building an automated timesheet system integrating Twilio SMS/Voice API.
- Developed Inventory, POS, and Digital Wallet APIs for a multi-tenant SaaS platform, ensuring strict data isolation and RBAC for 50+ enterprise clients.

## Certifications & Credentials

- AI Engineer Agentic Track: The Complete Agent & MCP Course — Udemy (Ed Donner / Ligency) · 17 hrs · March 2026
- AI Engineer Core Track: LLM Engineering, RAG, QLoRA, Agents — Udemy (Ed Donner / Ligency) · 33.5 hrs · March 2026
- AI Leader: Generative AI & Agentic AI for Leaders & Founders — Udemy (Ed Donner / Ligency) · 3.5 hrs · March 2026
- Andela AI Bootcamp — Agentic AI cohort · 2025–2026

## Education

Bachelor of Engineering (B.Eng.), Mechanical Engineering
University of Port Harcourt, Nigeria
