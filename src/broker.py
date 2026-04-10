from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from config import settings
from loguru import logger

class PaperTrader:
    def __init__(self):
        # Initialize Alpaca Client in PAPER mode
        self.client = TradingClient(
            settings.ALPACA_API_KEY, 
            settings.ALPACA_SECRET_KEY, 
            paper=True
        )
        logger.info("💼 Alpaca Paper Trader initialized.")

    def get_account_sync_data(self):
        """Fetches real-time cash and equity to sync with Supabase."""
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.buying_power),
                "equity": float(account.equity)
            }
        except Exception as e:
            logger.error(f"🚨 Failed to fetch Alpaca account: {e}")
            return None

    def execute_trade(self, ticker: str, action_type: int, size: float, current_price: float):
        """
        Executes the trade on Alpaca.
        Returns True if the order was submitted, False otherwise.
        """
        if action_type == 0:
            return False # HOLD

        # 🔥 FIX: Actually log why a trade is rejected due to 0 size!
        if size <= 0:
            logger.warning(f"⚠️ Trade rejected for {ticker}: Agent requested a trade size of {size}")
            return False

        try:
            if action_type == 1:  # BUY
                account = self.client.get_account()
                buying_power = float(account.buying_power)
                
                # Calculate how much cash to risk based on the RL Agent's Kelly size
                dollar_amount_to_risk = buying_power * size
                qty_to_buy = int(dollar_amount_to_risk // current_price)

                if qty_to_buy > 0:
                    order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=qty_to_buy,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    self.client.submit_order(order_data=order_data)
                    logger.success(f"🟢 EXECUTED BUY: {qty_to_buy} shares of {ticker} at ~${current_price}")
                    return True
                else:
                    logger.warning(f"⚠️ Insufficient funds to buy {ticker}.")
                    return False

            elif action_type == 2:  # SELL
                # Verify we own it before selling
                try:
                    position = self.client.get_open_position(ticker)
                    qty_to_sell = position.qty
                    
                    order_data = MarketOrderRequest(
                        symbol=ticker,
                        qty=qty_to_sell,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    self.client.submit_order(order_data=order_data)
                    logger.success(f"🔴 EXECUTED SELL: Liquidated {qty_to_sell} shares of {ticker}")
                    return True
                except Exception:
                    logger.warning(f"⚠️ Agent tried to sell {ticker}, but no open position exists.")
                    return False

        except Exception as e:
            logger.error(f"🚨 Broker execution failed for {ticker}: {e}")
            return False