import hashlib
import json
import time
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

# Import our verified core layers
from layer_1_proxy_guard import EnterpriseLayer1Guardrail
from layer_3_compliance_judge import EnterpriseLayer3Evaluator
from layer_4_deterministic_gateway import DeterministicGateway

# ==========================================
# 1. SAAS API PROTOCOLS & CONTRACTS (Pydantic V2 Optimized)
# ==========================================

class AgentExecutionRequest(BaseModel):
    transaction_type: str = Field(..., pattern="^(TRADE|CREDIT)$")
    raw_user_prompt: str = Field(..., json_schema_extra={"example": "Execute an automated buy for 100 shares of AAPL at $150."})
    proposed_payload: Dict[str, Any] = Field(..., json_schema_extra={"example": {
        "action_type": "BUY",
        "ticker": "AAPL",
        "quantity": 100,
        "price_limit": 150.0
    }})
    variables_used: List[str] = Field(default_factory=list, json_schema_extra={"example": ["credit_score", "debt_to_income"]})
    status: str = Field("APPROVED", pattern="^(APPROVED|DENIED)$")
    denial_rationale_text: str = Field(default="")

class GlobalAuditTrailEntry(BaseModel):
    timestamp: float
    request_id: str
    parent_hash: str
    risk_score: float
    sanitized_prompt: str
    compliance_passed: bool
    gateway_passed: bool
    cryptographic_hash: str

# ==========================================
# 2. INITIALIZE UNIFIED PROXY SAAS ENGINE
# ==========================================

app = FastAPI(
    title="Guardrails-as-a-Service (GaaS) API Proxy",
    description="Deterministic security and regulatory proxy middleware for autonomous financial agents.",
    version="1.0.0"
)

# Shared in-memory initialization simulating our SaaS infrastructure cache
firm_limits = {
    "max_single_trade_notional": 50000.00,
    "max_autonomous_credit_limit": 25000.00,
    "max_failure_threshold": 3
}

layer_1 = EnterpriseLayer1Guardrail(regional_profile="US")
layer_3 = EnterpriseLayer3Evaluator(compliance_profile="Consumer_Lending")
layer_4 = DeterministicGateway(firm_limits=firm_limits)

immutable_ledger: List[GlobalAuditTrailEntry] = []

# ==========================================
# 3. HELPER: CRYPTOGRAPHIC LINEAGE CHAINING
# ==========================================
def generate_block_hash(payload_dict: Dict[str, Any]) -> str:
    """Generates a SHA-256 hash representing an unalterable audit block state."""
    serialized_string = json.dumps(payload_dict, sort_keys=True)
    return hashlib.sha256(serialized_string.encode('utf-8')).hexdigest()

# ==========================================
# 4. EXPOSED API PROXY ENDPOINT
# ==========================================

@app.post("/api/v1/validate-agent", response_model=Dict[str, Any])
async def proxy_validate_agent(
    request: AgentExecutionRequest,
    x_api_key: str = Header(..., description="The financial institution's identifying key")
):
    start_time = time.time()
    request_id = f"tx_{int(start_time * 1000)}"
    
    # ─── STAGE 1: LAYER 1 INPUT GUARDRAIL ───
    l1_verdict = layer_1.verify_input_payload(request.raw_user_prompt)
    if not l1_verdict.is_safe:
        raise HTTPException(
            status_code=400, 
            detail={"error": "SECURITY_DROP", "message": "Input prompt triggered security filters.", "risk_score": l1_verdict.calculated_risk_score}
        )

    # ─── STAGE 2: LAYER 3 SEMANTIC POLICY JUDGE ───
    agent_payload_metadata = request.proposed_payload.copy()
    agent_payload_metadata["variables_used"] = request.variables_used
    agent_payload_metadata["status"] = request.status
    agent_payload_metadata["denial_rationale_text"] = request.denial_rationale_text

    l3_verdict = layer_3.evaluate_agent_proposal(agent_payload_metadata)
    if not l3_verdict.is_compliant:
        raise HTTPException(
            status_code=422,
            detail={"error": "REGULATORY_COMPLIANCE_VIOLATION", "reasons": l3_verdict.rejection_reasons, "rationale": l3_verdict.audit_rationale}
        )

    # ─── STAGE 3: LAYER 4 DETERMINISTIC GATEWAY ───
    try:
        validated_contract = layer_4.validate_and_route(request.transaction_type, request.proposed_payload)
    except Exception as e:
        raise HTTPException(
            status_code=403,
            detail={"error": "GATEWAY_ISOLATION_REJECTION", "message": str(e)}
        )

    # ─── STAGE 4: LAYER 5 IMMUTABLE EVENT-CHAINED LOGGING ───
    # Dynamically extract the hash of the preceding record to chain them together
    previous_hash = "GENESIS_BLOCK_0000000000000000" if not immutable_ledger else immutable_ledger[-1].cryptographic_hash

    log_payload = {
        "request_id": request_id,
        "sanitized_input": l1_verdict.sanitized_input,
        "risk_score": l1_verdict.calculated_risk_score,
        "validated_payload": request.proposed_payload,
        "timestamp": start_time,
        "parent_link_hash": previous_hash
    }
    crypto_hash = generate_block_hash(log_payload)
    
    audit_block = GlobalAuditTrailEntry(
        timestamp=start_time,
        request_id=request_id,
        parent_hash=previous_hash,
        risk_score=l1_verdict.calculated_risk_score,
        sanitized_prompt=l1_verdict.sanitized_input,
        compliance_passed=True,
        gateway_passed=True,
        cryptographic_hash=crypto_hash
    )
    immutable_ledger.append(audit_block)

    execution_latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "status": "AUTHORIZED_FOR_EXECUTION",
        "request_id": request_id,
        "execution_latency_ms": execution_latency_ms,
        "validated_payload": validated_contract.model_dump(),
        "cryptographic_receipt": crypto_hash
    }

@app.get("/api/v1/audit-ledger", response_model=List[GlobalAuditTrailEntry])
async def get_audit_ledger():
    """Returns the immutable compliance log for institutional validation."""
    return immutable_ledger

# ==========================================
# 5. LOCAL RUN ENVIRONMENT
# ==========================================
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Production GaaS Proxy Middleware Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)