//+------------------------------------------------------------------+
//| XAUUSD AI Trading Strategy EA (MQL5 - v6.0 TREND-RIDE)           |
//| Timeframe: M15 | Asset: XAUUSD                                   |
//|                                                                  |
//| CHANGE vs v5.0 (the ONLY change that beat the original in the    |
//| last 3 windows of every span): the fixed +5% take-profit is      |
//| REMOVED. Winners now ride a 3% trailing stop instead of being    |
//| capped. Backtest 2010-2026 (real XAUUSD 15m, gross of spread):   |
//|   2024-25 +22.8% vs +18.8% | 2023-25 +27.3% vs +22.8% |         |
//|   2022-26 +25.7% vs +22.8% (won all 6 recent windows on return). |
//| Tradeoff: ~2% more drawdown in some windows (winners give back   |
//| more on reversal). Entries/sizing/guardrails are UNCHANGED.      |
//+------------------------------------------------------------------+
#property copyright "AI Trading Strategy - Trend Ride"
#property link      "https://github.com"
#property version   "6.0"
#property strict

#include <Trade\Trade.mqh>

//--- Base Risk & Target Settings
input double RiskPercentage       = 0.5;    // Risk per trade as % of account
input double StopLossPercent      = 2.0;    // Initial stop loss %
input bool   UseTrailingOnly      = true;   // v6: TRUE = no fixed TP, ride the trend
input double TakeProfitFactor     = 2.5;    // (only used if UseTrailingOnly=false)
input double BuySignalThreshold   = 0.50;   // Minimum strength to enter
input int    MaxOpenTrades        = 1;      // ANTI-GRID: max simultaneous positions

//--- Funded Account Guardrails
input double DailyEquityLossLimit = 280.0;  // HARD daily equity drawdown stop ($)
input int    MaxConsecutiveLosses = 2;      // Stop for the day after X losses in a row
input int    TradeHourStart       = 2;      // Start trading (server time)
input int    TradeHourEnd         = 10;     // End trading (server time)

//--- Trade Management (v6: trailing is the primary exit)
input bool   EnableBreakEven      = false;  // v6 default OFF (matches the tested variant)
input double BE_TriggerPercent    = 1.0;    // Move to BE when +this% (if enabled)
input bool   EnableTrailingStop   = true;   // Trail the stop (the exit engine now)
input double TrailingStartPercent = 3.0;    // v6: start trailing at +3%
input double TrailingBufferPercent= 3.0;    // v6: trail 3% behind the peak (rides trends)

input bool   EnableDebugLogging   = true;

//--- Indicator handles
int handle_RSI, handle_MACD, handle_ATR, handle_ADX, handle_BB;
int handle_SMA20, handle_SMA50, handle_EMA12, handle_EMA26;

CTrade trade;

struct IndicatorValues {
    double rsi, macd, macd_signal, atr, adx;
    double sma20, sma50, ema12, ema26;
    double bb_upper, bb_middle, bb_lower;
};

//--- forward declarations
void CloseAllPositions();
void ManageOpenPositions();
bool CheckEquityStop();
bool HitMaxDailyLosses();
bool IsAllowedTradingTime();
bool IsAutoTradingAllowed();
bool FillIndicatorValues(IndicatorValues &ind);
bool CalculateBuySignal(IndicatorValues &ind, double &strength);
int  CountOpenTrades();
void ExecuteBuyOrder();

//+------------------------------------------------------------------+
int OnInit() {
    if(!IsAutoTradingAllowed()) {
        Print("INIT FAILED: Auto trading permissions are missing.");
        return INIT_FAILED;
    }
    handle_RSI   = iRSI(_Symbol, PERIOD_M15, 14, PRICE_CLOSE);
    handle_MACD  = iMACD(_Symbol, PERIOD_M15, 12, 26, 9, PRICE_CLOSE);
    handle_ATR   = iATR(_Symbol, PERIOD_M15, 14);
    handle_ADX   = iADX(_Symbol, PERIOD_M15, 14);
    handle_BB    = iBands(_Symbol, PERIOD_M15, 20, 0, 2.0, PRICE_CLOSE);
    handle_SMA20 = iMA(_Symbol, PERIOD_M15, 20, 0, MODE_SMA, PRICE_CLOSE);
    handle_SMA50 = iMA(_Symbol, PERIOD_M15, 50, 0, MODE_SMA, PRICE_CLOSE);
    handle_EMA12 = iMA(_Symbol, PERIOD_M15, 12, 0, MODE_EMA, PRICE_CLOSE);
    handle_EMA26 = iMA(_Symbol, PERIOD_M15, 26, 0, MODE_EMA, PRICE_CLOSE);

    trade.SetExpertMagicNumber(12345);
    trade.SetDeviationInPoints(30);

    if(handle_RSI == INVALID_HANDLE || handle_MACD == INVALID_HANDLE) return INIT_FAILED;
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
void OnTick() {
    if(!IsAutoTradingAllowed()) return;
    ManageOpenPositions();
    if(CheckEquityStop()) return;
    if(HitMaxDailyLosses()) return;

    static datetime last_bar_time = 0;
    datetime current_bar_time = iTime(_Symbol, PERIOD_M15, 0);
    if(current_bar_time == last_bar_time) return;
    if(!IsAllowedTradingTime()) return;
    last_bar_time = current_bar_time;

    IndicatorValues indicators;
    if(!FillIndicatorValues(indicators)) return;

    double buy_strength = 0;
    if(CalculateBuySignal(indicators, buy_strength) && CountOpenTrades() < MaxOpenTrades) {
        ExecuteBuyOrder();
    }
}

//+------------------------------------------------------------------+
bool IsAutoTradingAllowed() {
    if(!TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)) {
        if(EnableDebugLogging) Print("ERROR: Algo Trading disabled in the terminal.");
        return false;
    }
    if(!AccountInfoInteger(ACCOUNT_TRADE_EXPERT)) {
        if(EnableDebugLogging) Print("ERROR: EA trading disabled by broker for this account.");
        return false;
    }
    if(!AccountInfoInteger(ACCOUNT_TRADE_ALLOWED)) {
        if(EnableDebugLogging) Print("ERROR: Trading disabled for this account (read-only).");
        return false;
    }
    if(SymbolInfoInteger(_Symbol, SYMBOL_TRADE_MODE) == SYMBOL_TRADE_MODE_DISABLED) {
        if(EnableDebugLogging) Print("ERROR: Trading disabled for symbol ", _Symbol);
        return false;
    }
    return true;
}

//+------------------------------------------------------------------+
//| Trade Management: v6 trailing-stop-only exit (rides the trend)   |
//+------------------------------------------------------------------+
void ManageOpenPositions() {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == _Symbol) {
            double openPrice    = PositionGetDouble(POSITION_PRICE_OPEN);
            double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
            double currentSL    = PositionGetDouble(POSITION_SL);
            double currentTP    = PositionGetDouble(POSITION_TP);
            long   posType      = PositionGetInteger(POSITION_TYPE);

            double min_stop = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
            if(min_stop == 0) min_stop = 50 * SymbolInfoDouble(_Symbol, SYMBOL_POINT);

            if(posType == POSITION_TYPE_BUY) {
                // Optional break-even (default OFF in v6)
                if(EnableBreakEven) {
                    double beTrig = openPrice * (1.0 + BE_TriggerPercent / 100.0);
                    if(currentPrice >= beTrig && currentSL < openPrice) {
                        double newSL = openPrice * (1.0 + 0.05 / 100.0);
                        if(newSL - currentSL > min_stop)
                            trade.PositionModify(ticket, newSL, currentTP);
                    }
                }
                // Trailing stop = the exit engine. Activates at +Start%, holds Buffer% behind the peak.
                if(EnableTrailingStop) {
                    double trailTrig = openPrice * (1.0 + TrailingStartPercent / 100.0);
                    if(currentPrice >= trailTrig) {
                        double newSL = currentPrice * (1.0 - TrailingBufferPercent / 100.0);
                        if(newSL > currentSL && (currentPrice - newSL) >= min_stop)
                            trade.PositionModify(ticket, newSL, currentTP); // currentTP is 0 in v6
                    }
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
bool HitMaxDailyLosses() {
    MqlDateTime dt; TimeCurrent(dt);
    dt.hour = 0; dt.min = 0; dt.sec = 0;
    datetime day_start = StructToTime(dt);
    int loss_count = 0;
    if(HistorySelect(day_start, TimeCurrent())) {
        int total = HistoryDealsTotal();
        for(int i = total - 1; i >= 0; i--) {
            ulong ticket = HistoryDealGetTicket(i);
            if(ticket > 0 && HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
                double pnl = HistoryDealGetDouble(ticket, DEAL_PROFIT)
                           + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                           + HistoryDealGetDouble(ticket, DEAL_SWAP);
                if(pnl < 0) loss_count++; else break;
            }
        }
    }
    if(loss_count >= MaxConsecutiveLosses) {
        static datetime last_alert = 0;
        if(TimeCurrent() - last_alert > 3600) {
            Print("TRADING HALTED: ", MaxConsecutiveLosses, " consecutive losses today.");
            last_alert = TimeCurrent();
        }
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
bool CheckEquityStop() {
    MqlDateTime dt; TimeCurrent(dt);
    dt.hour = 0; dt.min = 0; dt.sec = 0;
    datetime day_start = StructToTime(dt);
    double todaysClosedPnL = 0;
    if(HistorySelect(day_start, TimeCurrent())) {
        int total = HistoryDealsTotal();
        for(int i = 0; i < total; i++) {
            ulong ticket = HistoryDealGetTicket(i);
            if(ticket > 0 && HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
                todaysClosedPnL += HistoryDealGetDouble(ticket, DEAL_PROFIT)
                                 + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
                                 + HistoryDealGetDouble(ticket, DEAL_SWAP);
            }
        }
    }
    double startBal = AccountInfoDouble(ACCOUNT_BALANCE) - todaysClosedPnL;
    double curEq    = AccountInfoDouble(ACCOUNT_EQUITY);
    if(startBal - curEq >= DailyEquityLossLimit) {
        CloseAllPositions();
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
void CloseAllPositions() {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionSelectByTicket(ticket) && PositionGetString(POSITION_SYMBOL) == _Symbol)
            trade.PositionClose(ticket);
    }
}

//+------------------------------------------------------------------+
bool IsAllowedTradingTime() {
    MqlDateTime dt; TimeCurrent(dt);
    if((dt.hour == 23 && dt.min >= 50) || (dt.hour == 0) || (dt.hour == 1 && dt.min <= 15)) return false;
    if(dt.hour < TradeHourStart || dt.hour >= TradeHourEnd) return false;
    if(dt.day_of_week == 4) return false;
    return true;
}

//+------------------------------------------------------------------+
bool FillIndicatorValues(IndicatorValues &ind) {
    double rsi[], macd_m[], macd_s[], atr[], adx[], bbu[], bbm[], bbl[], s20[], s50[], e12[], e26[];
    int count = 2;
    if(CopyBuffer(handle_RSI, 0, 0, count, rsi) < 0) return false;
    ArraySetAsSeries(rsi, true); ind.rsi = rsi[0];
    CopyBuffer(handle_MACD, 0, 0, count, macd_m); ArraySetAsSeries(macd_m, true); ind.macd = macd_m[0];
    CopyBuffer(handle_MACD, 1, 0, count, macd_s); ArraySetAsSeries(macd_s, true); ind.macd_signal = macd_s[0];
    CopyBuffer(handle_ATR, 0, 0, count, atr); ArraySetAsSeries(atr, true); ind.atr = atr[0];
    CopyBuffer(handle_ADX, 0, 0, count, adx); ArraySetAsSeries(adx, true); ind.adx = adx[0];
    CopyBuffer(handle_BB, 1, 0, count, bbu); ArraySetAsSeries(bbu, true); ind.bb_upper = bbu[0];
    CopyBuffer(handle_BB, 0, 0, count, bbm); ArraySetAsSeries(bbm, true); ind.bb_middle = bbm[0];
    CopyBuffer(handle_BB, 2, 0, count, bbl); ArraySetAsSeries(bbl, true); ind.bb_lower = bbl[0];
    CopyBuffer(handle_SMA20, 0, 0, count, s20); ArraySetAsSeries(s20, true); ind.sma20 = s20[0];
    CopyBuffer(handle_SMA50, 0, 0, count, s50); ArraySetAsSeries(s50, true); ind.sma50 = s50[0];
    CopyBuffer(handle_EMA12, 0, 0, count, e12); ArraySetAsSeries(e12, true); ind.ema12 = e12[0];
    CopyBuffer(handle_EMA26, 0, 0, count, e26); ArraySetAsSeries(e26, true); ind.ema26 = e26[0];
    return true;
}

//+------------------------------------------------------------------+
//| Entry signal — UNCHANGED from v5.0                               |
//+------------------------------------------------------------------+
bool CalculateBuySignal(IndicatorValues &ind, double &strength) {
    strength = 0;
    double price = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    if(ind.rsi < 70 && ind.rsi > 35) strength += 0.15;
    if(ind.macd > ind.macd_signal)   strength += 0.20;
    if(price > ind.sma20 && ind.ema12 > ind.ema26) strength += 0.25;
    if(ind.adx > 25.0)               strength += 0.15;
    if(price > ind.bb_middle)        strength += 0.25;
    return (strength >= BuySignalThreshold);
}

//+------------------------------------------------------------------+
int CountOpenTrades() {
    int count = 0;
    for(int i = PositionsTotal() - 1; i >= 0; i--)
        if(PositionGetSymbol(i) == _Symbol) count++;
    return count;
}

//+------------------------------------------------------------------+
//| Execute buy — v6: NO fixed take-profit (TP=0), trailing exits    |
//+------------------------------------------------------------------+
void ExecuteBuyOrder() {
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double sl  = NormalizeDouble(ask * (1.0 - StopLossPercent / 100.0), _Digits);
    double tp  = UseTrailingOnly ? 0.0
               : NormalizeDouble(ask + ((ask - sl) * TakeProfitFactor), _Digits);

    double risk_amount = AccountInfoDouble(ACCOUNT_BALANCE) * (RiskPercentage / 100.0);
    double tick_val = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    if(tick_size == 0 || tick_val == 0) return;

    double lot = risk_amount / (((ask - sl) / tick_size) * tick_val);
    double lot_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    lot = MathFloor(lot / lot_step) * lot_step;
    if(lot < SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);

    trade.Buy(lot, _Symbol, ask, sl, tp, "AI Buy Order v6");
}
//+------------------------------------------------------------------+
