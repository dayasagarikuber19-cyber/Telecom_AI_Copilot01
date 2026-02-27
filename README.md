Project Overview:
      ZENDS AI Customer Support Copilot is a Retrieval-Augmented Generation (RAG) based system designed for telecom customer support automation. The project combines intent classification across five categories (Billing, Complaint, Product Inquiry, Refund, Technical) with sentiment analysis and document-grounded retrieval to provide accurate responses based on official company documentation.

System Architecture:

     The system first analyzes customer queries by predicting intent and sentiment. It then retrieves relevant information from telecom policy and pricing PDFs using sentence embeddings and FAISS-based semantic similarity search to ensure responses are strictly grounded in company documents.

Key Features:

    The application supports country-specific pricing retrieval, refund policy queries, SLA details, contract terms, and telecom service information. Responses are generated based only on verified document content to avoid hallucinations and maintain answer reliability.

Human-in-the-Loop Validation:

     All AI-generated responses are subject to final human review to ensure accuracy, compliance, and quality before being delivered to customers. This approach enhances trust and maintains enterprise-level service standards.

Conclusion:

     This project demonstrates a practical implementation of a document-grounded AI assistant for telecom support. It highlights the effectiveness of Retrieval-Augmented Generation in delivering accurate, explainable, and scalable customer service automation.



