---
url: https://www.dinyangetoh.com/blog/event-driven-microservices-nestjs
title: Building Event-Driven Microservices with NestJS and AWS SNS/SQS | David Inyang-Etoh
path_segments:
- blog
- event-driven-microservices-nestjs
breadcrumb: Blog > Event Driven Microservices Nestjs
crawled_at: '2026-03-31T22:34:49.440064'
---

# Building Event-Driven Microservices with NestJS and AWS SNS/SQS | David Inyang-Etoh

Back to Blog Architecture 8 min read March 10, 2026

## Building Event-Driven Microservices with NestJS and AWS SNS/SQS

A deep dive into architecting scalable, fault-tolerant systems using NestJS, AWS SNS, SQS, and the Saga pattern for distributed transactions. NestJS AWS Microservices Architecture

## Building Event-Driven Microservices with NestJS and AWS SNS/SQS

Event-driven architecture is a powerful pattern for building scalable, loosely coupled systems. In this post, I'll walk you through how I architected a global payroll SaaS with 8 event-driven microservices at Collectwire.

## Why Event-Driven Architecture?

Traditional REST-based microservices create tight coupling between services. When Service A calls Service B synchronously, a failure in B cascades to A. Event-driven architecture breaks this coupling using a message broker as an intermediary.

## Setting Up AWS SNS and SQS

AWS SNS (Simple Notification Service) acts as the event publisher, while SQS (Simple Queue Service) provides durable, at-least-once delivery guarantees.

## Implementing the Saga Pattern

For distributed transactions spanning multiple services, we use the Saga pattern — a sequence of local transactions where each step publishes events to trigger the next.

## Key Takeaways

Idempotency is critical : Design your event handlers to be idempotent Dead Letter Queues : Always configure DLQs for failed message handling Observability : Correlate events with trace IDs across services Schema Registry : Use a schema registry to manage event contracts The result? 99.9% uptime, 3× delivery velocity, and infrastructure that scales effortlessly. David Inyang-Etoh Lead Software Engineer Share Article More Articles All posts 6 min read

#### How I Reduced AWS Costs by 66% Using CDK and Right-Sizing

5 min read

#### Integrating OpenAI into FinTech: Payment Summaries & Risk Insights

$ /$
