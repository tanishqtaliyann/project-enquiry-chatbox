import dotenv
import uvicorn 
import re
import json
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uuid
import asyncio
dotenv.load_dotenv()

# Pydantic Models
class StartRequest(BaseModel):
    message:str

class ContinueRequest(BaseModel):
    conversation_id:str
    answer:str

class ApiResponse(BaseModel):
    conversation_id:Optional[str] = None
    question:Optional[str]=None
    refined_query:Optional[str]=None

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://inquiry-system-sable.vercel.app",
    "https://juli-lockable-contentedly.ngrok-free.dev",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Inquiry System API"}

llm = ChatOpenAI(model="gpt-4o", temperature=0.8)

conversations_db: Dict[str, List[Any]] = {}

SYSTEM_PROMPT = """
You are a helpful, friendly person helping someone refine their request. Talk naturally, like you're having a real conversation.

CONVERSATION FLOW (STRICTLY FOLLOW):
1. FIRST QUESTION - Ask about their role: "What is your role? Are you a student, working professional, researcher, or something else?"
2. QUESTIONS 2-4 - Ask EXACTLY 3 clarifying questions about their problem, building on their previous answers
3. AFTER 4TH QUESTION - You MUST output the final refined query

CRITICAL RULES:
- Start with the role question, then ask exactly 3 problem clarifications = 4 questions total
- Ask one clarifying question at a time, building on their previous answers
- Use natural acknowledgments: "Got it", "Nice", "Understood", "Great", "I understand" - keep it brief and genuine
- When someone seems distressed, stuck, or frustrated, console them first. Say something like "I understand" or "I hear you" to acknowledge their feelings before asking your question
- Include helpful examples in your questions when it makes sense (e.g., "For example, Python + FastAPI, Node.js, or Java Spring?")
- Keep responses conversational and brief - don't overthink or be overly formal
- After the 4th question (1 role + 3 clarifying), you MUST stop asking and output the final refined query

When outputting the final query:
- Start with the phrase: "Here's your refined query:"
- MUST include the user's role/profession at the beginning of the query
- Format: Here's your refined query: As a [role], [user's refined request with all context]
- Example: Here's your refined query: As a student learning web development, I need help debugging a React component that isn't rendering properly
- Do NOT add "Hope this helps!" or similar closing statements after the query
- The phrase "Here's your refined query:" is REQUIRED at the beginning of the final output

Write like a real person would talk - natural, warm, and helpful. Avoid sounding like a robot or following a rigid script.
"""


@app.post("/inquire/start/stream")
async def start_inquiry_stream(request:StartRequest):
    conversation_id = str(uuid.uuid4())

    history = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request.message)
    ]
    
    # Store conversation immediately to prevent race conditions
    # We'll update it after streaming completes
    conversations_db[conversation_id] = history

    async def generate():
        full_content = ""
        found_final_query = False
        content_before_final = ""
        
        async for chunk in llm.astream(history):
            if chunk.content:
                full_content += chunk.content
                
                # Check if we detected "Here's your refined query:" phrase
                if not found_final_query:
                    if "here's your refined query:" in full_content.lower():
                        found_final_query = True
                        # Extract content before the phrase (should be empty per instructions, but just in case)
                        prefix_match = re.search(r"here's your refined query:\s*", full_content, re.IGNORECASE)
                        if prefix_match:
                            content_before_final = full_content[:prefix_match.start()].strip()
                        # Stop sending tokens to frontend immediately
                        # But continue accumulating all remaining chunks
                    else:
                        # Still streaming normally, send tokens
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk.content, 'conversation_id': conversation_id})}\n\n"
                # If found_final_query is True, we're accumulating but not sending
        
        # After all chunks are received, extract the complete final query
        if found_final_query:
            prefix_match = re.search(r"here's your refined query:\s*", full_content, re.IGNORECASE)
            if prefix_match:
                query_start = prefix_match.end()
                query_text = full_content[query_start:].strip()
                
                # Extract everything after "Here's your refined query:" until double newline (paragraph break) or end
                # This captures the full query even if it spans multiple lines
                query_lines = query_text.split('\n\n')
                if query_lines:
                    # If there's a double newline, take everything before it (the query)
                    query = query_lines[0].strip()
                else:
                    # No double newline, take everything but clean up trailing single newlines
                    # Remove trailing newlines but keep the content
                    query = query_text.rstrip('\n').strip()
                
                # Remove any trailing phrases like "Hope this helps!" that might be on same line
                # Look for common closing phrases and remove them
                closing_phrases = ['hope this helps', 'does that help', 'hope that helps', 'let me know', 'hope this', 'does that']
                query_lower = query.lower()
                for phrase in closing_phrases:
                    if phrase in query_lower:
                        # Find and remove the phrase and everything after it
                        idx = query_lower.find(phrase)
                        query = query[:idx].strip()
                        break
                
                if query:
                    # Only delete if conversation exists (should always exist, but safe check)
                    if conversation_id in conversations_db:
                        del conversations_db[conversation_id]
                    formatted_query = f"User wants to say this: {query}"
                    yield f"data: {json.dumps({'type': 'final_query', 'refined_query': formatted_query})}\n\n"
                    return
        
        # If not a final query, update stored conversation and send done message
        ai_message = AIMessage(content=full_content)
        history.append(ai_message)
        conversations_db[conversation_id] = history
        
        # Send final message
        yield f"data: {json.dumps({'type': 'done', 'conversation_id': conversation_id, 'question': full_content})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/inquire/start", response_model=ApiResponse)
async def start_inquiry(request:StartRequest):
    conversation_id = str(uuid.uuid4())

    history = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request.message)
    ]

    response = llm.invoke(history)
    
    history.append(response)

    conversations_db[conversation_id] = history

    return ApiResponse(conversation_id=conversation_id, question=response.content)

@app.post("/inquire/continue/stream")
async def continue_inquiry_stream(request:ContinueRequest):
    print(f"Continue request - conversation_id: {request.conversation_id}")
    print(f"Available conversations: {list(conversations_db.keys())}")
    
    if request.conversation_id not in conversations_db:
        async def error_generate():
            yield f"data: {json.dumps({'type': 'error', 'content': 'Conversation not found.'})}\n\n"
        return StreamingResponse(error_generate(), media_type="text/event-stream")
    
    try:
        # Get history - check again in case it was deleted between check and access
        if request.conversation_id not in conversations_db:
            async def error_generate():
                yield f"data: {json.dumps({'type': 'error', 'content': 'Conversation not found.'})}\n\n"
            return StreamingResponse(error_generate(), media_type="text/event-stream")
        
        history = list(conversations_db[request.conversation_id])  # Create a new list
        history.append(HumanMessage(content=request.answer))

        async def generate():
            full_content = ""
            found_final_query = False
            
            async for chunk in llm.astream(history):
                if chunk.content:
                    full_content += chunk.content
                    
                    # Check if we detected "Here's your refined query:" phrase
                    if not found_final_query:
                        if "here's your refined query:" in full_content.lower():
                            found_final_query = True
                            # Stop sending tokens to frontend immediately
                            # But continue accumulating all remaining chunks
                        else:
                            # Still streaming normally, send tokens
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content, 'conversation_id': request.conversation_id})}\n\n"
                    # If found_final_query is True, we're accumulating but not sending
            
            # After all chunks are received, extract the complete final query
            if found_final_query:
                prefix_match = re.search(r"here's your refined query:\s*", full_content, re.IGNORECASE)
                if prefix_match:
                    query_start = prefix_match.end()
                    query_text = full_content[query_start:].strip()
                    
                    # Extract everything after "Here's your refined query:" until double newline (paragraph break) or end
                    # This captures the full query even if it spans multiple lines
                    query_lines = query_text.split('\n\n')
                    if query_lines:
                        # If there's a double newline, take everything before it (the query)
                        query = query_lines[0].strip()
                    else:
                        # No double newline, take everything but clean up trailing single newlines
                        # Remove trailing newlines but keep the content
                        query = query_text.rstrip('\n').strip()
                    
                    # Remove any trailing phrases like "Hope this helps!" that might be on same line
                    # Look for common closing phrases and remove them
                    closing_phrases = ['hope this helps', 'does that help', 'hope that helps', 'let me know', 'hope this', 'does that']
                    query_lower = query.lower()
                    for phrase in closing_phrases:
                        if phrase in query_lower:
                            # Find and remove the phrase and everything after it
                            idx = query_lower.find(phrase)
                            query = query[:idx].strip()
                            break
                    
                    if query:
                        # Only delete if conversation exists
                        if request.conversation_id in conversations_db:
                            del conversations_db[request.conversation_id]
                        formatted_query = f"User wants to say this: {query}"
                        yield f"data: {json.dumps({'type': 'final_query', 'refined_query': formatted_query})}\n\n"
                        return
            
            # Continue conversation
            ai_message = AIMessage(content=full_content)
            history.append(ai_message)
            # Only update if conversation still exists (might have been deleted in race condition)
            if request.conversation_id in conversations_db:
                conversations_db[request.conversation_id] = history
            
            yield f"data: {json.dumps({'type': 'done', 'conversation_id': request.conversation_id, 'question': full_content})}\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        async def error_generate():
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
        return StreamingResponse(error_generate(), media_type="text/event-stream")

@app.post("/inquire/continue", response_model=ApiResponse)
async def continue_inquiry(request:ContinueRequest):
    if request.conversation_id not in conversations_db:
        return ApiResponse(refined_query="Error: Conversation not found.")
    
    try:
        history = conversations_db[request.conversation_id]
        history.append(HumanMessage(content=request.answer))

        response = llm.invoke(history)

        response_content = response.content
        if "here's your refined query:" in response_content.lower():
            # Extract query after "Here's your refined query:"
            match = re.search(r"here's your refined query:\s*(.+?)(?:\n\n|\n$|$)", response_content, re.IGNORECASE | re.DOTALL)
            if match:
                query = match.group(1).strip().split('\n')[0].strip()
                # Only delete if conversation exists
                if request.conversation_id in conversations_db:
                    del conversations_db[request.conversation_id]
                # Format as system reference
                formatted_query = f"User wants to say this: {query}"
                return ApiResponse(refined_query=formatted_query)
        
        # Continue conversation (either no final_query tag or extraction failed)
        history.append(response)
        conversations_db[request.conversation_id] = history
        return ApiResponse(conversation_id=request.conversation_id, question=response.content)
    except Exception as e:
        return ApiResponse(refined_query=f"Error: {str(e)}")

if __name__ =="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

