"""
execution.py — Live order execution via Groww API

Handles:
  • Market and limit order placement
  • Order status polling
  • Position tracking
  • Square-off (intraday close)

Usage:
    from login import generate_session
    from execution import Executor

    api = generate_session()
    ex = Executor(api, symbol="RELIANCE", exchange="NSE")
    ex.buy(quantity=10)
    ex.sell(quantity=10)
"""

import time


class Executor:
    def __init__(self, api, symbol, exchange="NSE", segment="CASH", dry_run=False):
        """
        Parameters
        ----------
        api       : authenticated GrowwAPI instance
        symbol    : e.g. "RELIANCE"
        exchange  : "NSE" or "BSE"
        segment   : "CASH" (equity) or "FNO"
        dry_run   : if True, log orders but do not send to Groww
        """
        self.api = api
        self.symbol = symbol
        self.exchange = exchange
        self.segment = segment
        self.dry_run = dry_run
        self.position = 0           # net open position (shares)
        self.orders = []            # log of all orders placed this session

    # ─────────────────────────────────────────────────────────────────────
    # Core order methods
    # ─────────────────────────────────────────────────────────────────────

    def buy(self, quantity, order_type="MARKET", price=None):
        """Place a buy order. Returns order ID or None on failure."""
        return self._place_order("BUY", quantity, order_type, price)

    def sell(self, quantity, order_type="MARKET", price=None):
        """Place a sell order. Returns order ID or None on failure."""
        return self._place_order("SELL", quantity, order_type, price)

    def square_off(self):
        """Close all open positions with a market order."""
        if self.position == 0:
            print("No open position to square off.")
            return
        qty = abs(self.position)
        direction = "SELL" if self.position > 0 else "BUY"
        print(f"🔴 Squaring off {qty} shares ({direction})")
        return self._place_order(direction, qty, "MARKET")

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _place_order(self, transaction_type, quantity, order_type, price=None):
        if quantity <= 0:
            print(f"⚠️  Skipping zero-quantity order.")
            return None

        order_params = {
            "groww_symbol": f"{self.exchange}-{self.symbol}",
            "exchange": self.exchange,
            "segment": self.segment,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": order_type,
            "product": "INTRADAY",    # always intraday to avoid overnight margin
        }
        if order_type == "LIMIT" and price is not None:
            order_params["price"] = price

        if self.dry_run:
            print(f"[DRY RUN] Would place order: {order_params}")
            self.orders.append({"params": order_params, "order_id": "DRY_RUN"})
            self._update_position(transaction_type, quantity)
            return "DRY_RUN"

        try:
            response = self.api.place_order(**order_params)
            order_id = response.get("order_id") or response.get("orderId")
            if order_id:
                print(f"✅ {transaction_type} order placed — ID: {order_id}, qty: {quantity}")
                self.orders.append({"params": order_params, "order_id": order_id})
                self._update_position(transaction_type, quantity)
                return order_id
            else:
                print(f"❌ Order placement failed. Response: {response}")
                return None
        except Exception as e:
            print(f"❌ Order error: {e}")
            return None

    def _update_position(self, transaction_type, quantity):
        if transaction_type == "BUY":
            self.position += quantity
        elif transaction_type == "SELL":
            self.position -= quantity

    def get_order_status(self, order_id, retries=5, delay=2):
        """Poll for order fill status with retries."""
        for attempt in range(retries):
            try:
                response = self.api.get_order_details(order_id=order_id)
                status = response.get("status", "UNKNOWN")
                print(f"  Order {order_id} — status: {status} (attempt {attempt+1})")
                if status in ("COMPLETE", "REJECTED", "CANCELLED"):
                    return status
                time.sleep(delay)
            except Exception as e:
                print(f"  Error polling order: {e}")
                time.sleep(delay)
        return "TIMEOUT"

    def get_ltp(self):
        """Fetch the last traded price for this symbol."""
        try:
            response = self.api.get_ltp(
                groww_symbol=f"{self.exchange}-{self.symbol}",
                exchange=self.exchange,
                segment=self.segment,
            )
            return response.get("ltp") or response.get("last_price")
        except Exception as e:
            print(f"Could not fetch LTP: {e}")
            return None

    def print_session_summary(self):
        """Print a summary of all orders placed in this session."""
        print(f"\n─── Session Summary for {self.symbol} ───")
        print(f"  Orders placed : {len(self.orders)}")
        print(f"  Net position  : {self.position} shares")
        for o in self.orders:
            print(f"  [{o['order_id']}] {o['params']['transaction_type']} "
                  f"{o['params']['quantity']} @ {o['params']['order_type']}")