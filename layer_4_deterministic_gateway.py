import time
from typing import Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError

# ==========================================
# 1. RIGID FINANCIAL TRANSACTION CONTRACTS (Pydantic V2 Optimized)
# ==========================================

class EnforcedTradeContract(BaseModel):
    action_type: str = Field(..., pattern="^(BUY|SELL)$")
    ticker: str = Field(..., min_length=1, max_length=5)
    quantity: int = Field(..., gt=0)
    price_limit: float = Field(..., gt=0.0)

class EnforcedCreditContract(BaseModel):
    action_type: str = Field(..., pattern="^(APPROVE_CREDIT)$")
    customer_id: str = Field(..., min_length=3)
    approved_limit: float = Field(..., gt=0.0)

# ==========================================
# 2. DETERMINISTIC CAGE & CIRCUIT BREAKER
# ==========================================

class DeterministicGateway:
    def __init__(self, firm_limits: Dict[str, Any]):
        self.limits = firm_limits
        self.consecutive_failures = 0
        self.circuit_broken = False
        self.last_failure_timestamp = 0.0

    def check_circuit_breaker(self) -> None:
        """Verifies if the systemic kill switch has been tripped."""
        if self.circuit_broken:
            # Cool-down window check: Allow automatic reset after 60 seconds of isolation
            if time.time() - self.last_failure_timestamp > 60:
                print("🔄 COOLDOWN EXPIRED: Resetting gateway circuit breaker to Closed state...")
                self.circuit_broken = False
                self.consecutive_failures = 0
            else:
                raise RuntimeError("🚨 CRITICAL: Circuit Breaker is OPEN. Autonomous execution pipeline is frozen.")

    def trip_breaker(self, reason: str) -> None:
        """Forces the immediate isolation of the agent network."""
        self.circuit_broken = True
        self.last_failure_timestamp = time.time()
        print(f"💥 CIRCUIT BREAKER TRIPPED: System entering emergency lockdown. Reason: {reason}")

    def validate_and_route(self, transaction_type: str, raw_agent_payload: Dict[str, Any]) -> BaseModel:
        """Enforces absolute limits, typing validity, and programmatic rule boundaries."""
        # Step 1: Check active state of the ecosystem
        self.check_circuit_breaker()

        # Step 2: Enforce Strict Structural Validation via Pydantic
        try:
            if transaction_type == "TRADE":
                validated_payload = EnforcedTradeContract(**raw_agent_payload)
                # Compute total notional exposure
                total_notional = validated_payload.quantity * validated_payload.price_limit
                
                # Check absolute corporate trading limit limits
                if total_notional > self.limits["max_single_trade_notional"]:
                    raise ValueError(f"Notional value ${total_notional} violates absolute cap (${self.limits['max_single_trade_notional']})")
                
            elif transaction_type == "CREDIT":
                validated_payload = EnforcedCreditContract(**raw_agent_payload)
                
                # Check absolute corporate underwriting line assignment limits
                if validated_payload.approved_limit > self.limits["max_autonomous_credit_limit"]:
                    raise ValueError(f"Line request ${validated_payload.approved_limit} violates absolute cap (${self.limits['max_autonomous_credit_limit']})")
            else:
                raise ValueError(f"Unsupported validation channel target: {transaction_type}")

            # Zero out any operational failures upon successful pipeline execution
            self.consecutive_failures = 0
            return validated_payload

        except (ValidationError, ValueError) as e:
            self.consecutive_failures += 1
            print(f"⚠️ Validation Fault Recorded ({self.consecutive_failures}/{self.limits['max_failure_threshold']}): {str(e)}")
            
            if self.consecutive_failures >= self.limits["max_failure_threshold"]:
                self.trip_breaker(reason="Consecutive faults exceeded threshold limit count.")
                
            raise SecurityException(f"GATEWAY_REJECTION: Transaction blocked. Details: {str(e)}")

class SecurityException(Exception):
    pass

# ==========================================
# 3. VERIFICATION MATRIX RUNTIMES
# ==========================================
if __name__ == "__main__":
    # Standard enterprise parameter profiles
    risk_config = {
        "max_single_trade_notional": 50000.00,       # Max $50k transaction value limit
        "max_autonomous_credit_limit": 25000.00,    # Max $25k autonomous loan assignment
        "max_failure_threshold": 2                   # Allow max 2 consecutive failures before lock
    }

    gateway = DeterministicGateway(firm_limits=risk_config)

    print("--- ⚙️ Verification 1: Safe Processing Under Gateway Thresholds ---")
    valid_trade = {"action_type": "BUY", "ticker": "AAPL", "quantity": 100, "price_limit": 150.0}
    tx_authorized = gateway.validate_and_route("TRADE", valid_trade)
    print(f"Authorized Action Contract Created: {tx_authorized.model_dump_json()}\n")

    print("--- ⚙️ Verification 2: Rogue Action Volatility Fault Generation ---")
    # Agent attempts a trade worth $150,000 (Limit is $50,000)
    massive_trade = {"action_type": "BUY", "ticker": "TSLA", "quantity": 100, "price_limit": 1500.0}
    try:
        gateway.validate_and_route("TRADE", massive_trade)
    except SecurityException as e:
        print(f"Result: {e}\n")

    print("--- ⚙️ Verification 3: Threshold Breached – Cascading Circuit Trip ---")
    # Generating consecutive structural payload mutation errors to force lockdown
    malformed_credit_payload = {"action_type": "APPROVE_CREDIT", "customer_id": "XY", "approved_limit": -5000.0}
    try:
        gateway.validate_and_route("CREDIT", malformed_credit_payload)
    except SecurityException as e:
        print(f"Result: {e}\n")

    print("--- ⚙️ Verification 4: Pipeline Execution While Circuit Breaker Is Open ---")
    try:
        gateway.validate_and_route("TRADE", valid_trade)
    except RuntimeError as e:
        print(f"Result: {e}\n")