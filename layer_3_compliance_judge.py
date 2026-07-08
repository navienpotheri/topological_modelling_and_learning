from typing import Dict, Any, Tuple, List
from pydantic import BaseModel, Field

class ComplianceEvaluationVerdict(BaseModel):
    is_compliant: bool = Field(..., description="Flags if the agent's proposal is legally and contextually clear to proceed.")
    rejection_reasons: List[str] = Field(default_factory=list, description="Specific regulatory or policy violation identifiers.")
    audit_rationale: str = Field(..., description="Detailed textual breakdown justifying the evaluation decision for compliance audit trails.")

class EnterpriseLayer3Evaluator:
    def __init__(self, compliance_profile: str = "Consumer_Lending"):
        self.compliance_profile = compliance_profile
        
        # Hard-coded array of forbidden bias variables (Proxies for discriminatory metrics)
        self.forbidden_demographic_proxies = {
            "zip_code", "postal_code", "neighborhood_tier", "gender", "race", "ethnicity", "religion"
        }

    def evaluate_agent_proposal(self, agent_output_payload: Dict[str, Any]) -> ComplianceEvaluationVerdict:
        violations = []
        rationales = []
        
        # --- RULE CHECK 1: Protected Variables & Latent Bias Detection (ECOA / Fair Lending) ---
        # Scan through the keys and data properties used by the agent to make its decision
        decision_variables = agent_output_payload.get("variables_used", [])
        for var in decision_variables:
            if str(var).lower() in self.forbidden_demographic_proxies:
                violations.append("FAIR_LENDING_BIAS_VIOLATION")
                rationales.append(f"Agent utilized a prohibited geographic/demographic attribute token: [{var}].")

        # --- RULE CHECK 2: Adverse Action Explainability Integrity ---
        # If the agent is issuing a denial, it MUST supply a valid, auditable textual reason
        if agent_output_payload.get("status") == "DENIED":
            explanation = agent_output_payload.get("denial_rationale_text", "").strip()
            
            # Catch vague, unhelpful explanations that violate CFPB transparency laws
            if not explanation or len(explanation) < 20 or "internal scoring" in explanation.lower():
                violations.append("EXPLAINABILITY_DEFICIT")
                rationales.append("Agent failed to provide an explicit, compliant Adverse Action justification statement.")

        # --- DECISION ROUTING ---
        is_compliant = len(violations) == 0
        final_rationale_summary = " ".join(rationales) if not is_compliant else "Contextual pass: No bias signals or semantic rule deviations detected."

        return ComplianceEvaluationVerdict(
            is_compliant=is_compliant,
            rejection_reasons=violations,
            audit_rationale=final_rationale_summary
        )

# ==========================================
# RUNTIME COMPLIANCE VERIFICATION VERDICTS
# ==========================================
if __name__ == "__main__":
    judge = EnterpriseLayer3Evaluator(compliance_profile="Consumer_Lending")

    print("--- ⚖️ Verification 1: Discriminatory Bias Catch ---")
    # Simulating a flawed loan evaluation agent payload using a forbidden geographic proxy
    bad_agent_output = {
        "action_type": "CREDIT_LINE_CALCULATION",
        "customer_id": "cust_90112",
        "status": "APPROVED",
        "proposed_limit": 15000.00,
        "variables_used": ["credit_score", "debt_to_income", "zip_code"] # <-- Violation
    }
    verdict_1 = judge.evaluate_agent_proposal(bad_agent_output)
    print(f"Is Compliant: {verdict_1.is_compliant}")
    print(f"Violations: {verdict_1.rejection_reasons}")
    print(f"Audit Trail Rationale: {verdict_1.audit_rationale}\n")

    print("--- ⚖️ Verification 2: Vague Denial Explainability Catch ---")
    # Agent denies credit, but fails to provide a legal, structured reason
    vague_agent_output = {
        "action_type": "LOAN_UNDERWRITING",
        "customer_id": "cust_4431",
        "status": "DENIED",
        "denial_rationale_text": "Denied based on internal blackbox calculation algorithms.", # <-- Violation
        "variables_used": ["credit_score", "debt_to_income"]
    }
    verdict_2 = judge.evaluate_agent_proposal(vague_agent_output)
    print(f"Is Compliant: {verdict_2.is_compliant}")
    print(f"Violations: {verdict_2.rejection_reasons}")
    print(f"Audit Trail Rationale: {verdict_2.audit_rationale}\n")