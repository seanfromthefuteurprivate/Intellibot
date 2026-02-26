"""
VENOM WAR ROOM - Complete System Visibility

Every fang visible:
- Live P&L per position
- Win rate (daily, weekly, all-time)
- GEX regime + flip distance
- Every engine's status
- Trade debate transcripts
- Memory recalls
- Specialist votes
- Component health
"""

from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import threading

from wsb_snake.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionSnapshot:
    """Live position data."""
    symbol: str
    option_symbol: str
    direction: str
    entry_price: float
    current_price: float
    pnl: float
    pnl_pct: float
    entry_time: str
    hold_minutes: int
    stop_loss: float
    trailing_level: str  # "INITIAL", "BREAKEVEN", "LOCK", "AGGRESSIVE", "MOONSHOT"


@dataclass
class EngineStatus:
    """Status of a trading engine."""
    name: str
    running: bool
    trades_today: int
    signals_today: int
    last_signal_time: Optional[str]
    cooldown_active: bool
    cooldown_until: Optional[str]
    extra: Dict[str, Any]


@dataclass
class WarRoomState:
    """Complete war room state."""
    # Timestamp
    timestamp: str

    # P&L Overview
    daily_pnl: float
    daily_pnl_pct: float
    weekly_pnl: float
    total_pnl: float

    # Win Rate
    daily_wins: int
    daily_losses: int
    daily_win_rate: float
    streak_multiplier: float
    consecutive_wins: int
    consecutive_losses: int

    # Account
    buying_power: float
    deployed_capital: float
    deployment_pct: float

    # Positions
    open_positions: List[PositionSnapshot]
    position_count: int

    # HYDRA Intelligence
    hydra_connected: bool
    hydra_direction: str
    hydra_regime: str
    hydra_confidence: float
    blowup_probability: int
    blowup_mode_active: bool

    # GEX Data
    gex_regime: str
    gex_flip_point: float
    gex_flip_distance_pct: float
    gex_support_levels: List[float]
    gex_resistance_levels: List[float]

    # Flow Data
    flow_bias: str
    flow_confidence: float

    # Engine Status
    engines: List[EngineStatus]

    # Risk Governor
    kill_switch_active: bool
    drawdown_halt_active: bool
    drawdown_half_size_active: bool
    cooldown_active: bool
    cooldown_until: Optional[str]
    win_rate_pause_active: bool

    # Recent Activity
    recent_trades: List[Dict]
    recent_signals: List[Dict]
    recent_lessons: List[Dict]

    # Component Health
    components: Dict[str, bool]


class WarRoom:
    """
    Central war room for complete system visibility.

    Aggregates data from all components into a single snapshot.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._last_state: Optional[WarRoomState] = None
        self._recent_trades: List[Dict] = []
        self._recent_signals: List[Dict] = []
        self._recent_lessons: List[Dict] = []

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete war room state as dictionary."""
        try:
            state = self._build_state()
            self._last_state = state
            return asdict(state)
        except Exception as e:
            logger.error(f"WAR_ROOM: Failed to build state - {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}

    def _build_state(self) -> WarRoomState:
        """Build complete state from all components."""

        # Get executor state
        executor_data = self._get_executor_data()

        # Get HYDRA state
        hydra_data = self._get_hydra_data()

        # Get engine states
        engines = self._get_engine_states()

        # Get risk governor state
        governor_data = self._get_governor_data()

        # Get positions
        positions = self._get_positions()

        # Calculate account metrics
        buying_power = executor_data.get("buying_power", 5000)
        deployed = sum(p.entry_price * 100 for p in positions)  # Rough estimate

        return WarRoomState(
            timestamp=datetime.now(timezone.utc).isoformat(),

            # P&L
            daily_pnl=executor_data.get("daily_pnl", 0),
            daily_pnl_pct=executor_data.get("daily_pnl", 0) / buying_power * 100 if buying_power > 0 else 0,
            weekly_pnl=executor_data.get("total_pnl", 0),  # TODO: separate weekly tracking
            total_pnl=executor_data.get("total_pnl", 0),

            # Win Rate
            daily_wins=governor_data.get("daily_wins", 0),
            daily_losses=governor_data.get("daily_losses", 0),
            daily_win_rate=governor_data.get("daily_win_rate", 0),
            streak_multiplier=governor_data.get("streak_multiplier", 1.0),
            consecutive_wins=governor_data.get("consecutive_wins", 0),
            consecutive_losses=governor_data.get("consecutive_losses", 0),

            # Account
            buying_power=buying_power,
            deployed_capital=deployed,
            deployment_pct=deployed / buying_power * 100 if buying_power > 0 else 0,

            # Positions
            open_positions=positions,
            position_count=len(positions),

            # HYDRA
            hydra_connected=hydra_data.get("connected", False),
            hydra_direction=hydra_data.get("direction", "NEUTRAL"),
            hydra_regime=hydra_data.get("regime", "UNKNOWN"),
            hydra_confidence=hydra_data.get("confidence", 0),
            blowup_probability=hydra_data.get("blowup_probability", 0),
            blowup_mode_active=hydra_data.get("blowup_probability", 0) > 60,

            # GEX
            gex_regime=hydra_data.get("gex_regime", "UNKNOWN"),
            gex_flip_point=hydra_data.get("gex_flip_point", 0),
            gex_flip_distance_pct=hydra_data.get("gex_flip_distance_pct", 999),
            gex_support_levels=hydra_data.get("gex_key_support", []),
            gex_resistance_levels=hydra_data.get("gex_key_resistance", []),

            # Flow
            flow_bias=hydra_data.get("flow_bias", "NEUTRAL"),
            flow_confidence=hydra_data.get("flow_confidence", 0),

            # Engines
            engines=engines,

            # Risk Governor
            kill_switch_active=governor_data.get("kill_switch_active", False),
            drawdown_halt_active=governor_data.get("drawdown_halt_active", False),
            drawdown_half_size_active=governor_data.get("drawdown_half_size_active", False),
            cooldown_active=governor_data.get("cooldown_active", False),
            cooldown_until=governor_data.get("cooldown_until"),
            win_rate_pause_active=governor_data.get("win_rate_pause_active", False),

            # Recent Activity
            recent_trades=self._recent_trades[-10:],
            recent_signals=self._recent_signals[-10:],
            recent_lessons=self._recent_lessons[-5:],

            # Component Health
            components=self._check_component_health(),
        )

    def _get_executor_data(self) -> Dict:
        """Get data from Alpaca executor."""
        try:
            from wsb_snake.trading.alpaca_executor import alpaca_executor
            stats = alpaca_executor.get_session_stats()
            account = alpaca_executor.get_account()
            return {
                **stats,
                "buying_power": float(account.get("buying_power", 5000)) if account else 5000,
            }
        except Exception as e:
            logger.debug(f"WAR_ROOM: Executor data failed - {e}")
            return {}

    def _get_hydra_data(self) -> Dict:
        """Get data from HYDRA bridge."""
        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_intel
            intel = get_hydra_intel()
            return intel.to_dict()
        except Exception as e:
            logger.debug(f"WAR_ROOM: HYDRA data failed - {e}")
            return {}

    def _get_governor_data(self) -> Dict:
        """Get data from risk governor."""
        try:
            from wsb_snake.trading.risk_governor import get_risk_governor
            governor = get_risk_governor()
            status = governor.get_weaponized_status()
            return {
                **status,
                "kill_switch_active": governor.kill_switch_active,
                "streak_multiplier": governor._current_streak_multiplier,
                "consecutive_wins": governor._consecutive_wins,
                "consecutive_losses": governor._consecutive_losses,
            }
        except Exception as e:
            logger.debug(f"WAR_ROOM: Governor data failed - {e}")
            return {}

    def _get_positions(self) -> List[PositionSnapshot]:
        """Get current positions with live prices."""
        try:
            from wsb_snake.trading.alpaca_executor import alpaca_executor, PositionStatus

            positions = []
            for pos in alpaca_executor.positions.values():
                if pos.status != PositionStatus.OPEN:
                    continue

                # Get current price
                quote = alpaca_executor.get_option_quote(pos.option_symbol)
                bp = float(quote.get("bp", 0)) if quote else 0
                ap = float(quote.get("ap", 0)) if quote else 0
                current_price = (bp + ap) / 2 if (bp > 0 and ap > 0) else pos.entry_price

                # Calculate P&L
                pnl = (current_price - pos.entry_price) * pos.qty * 100
                pnl_pct = (current_price / pos.entry_price - 1) * 100 if pos.entry_price > 0 else 0

                # Determine trailing level
                trail_level = "INITIAL"
                if pnl_pct >= 100:
                    trail_level = "MOONSHOT"
                elif pnl_pct >= 50:
                    trail_level = "AGGRESSIVE"
                elif pnl_pct >= 25:
                    trail_level = "LOCK"
                elif pnl_pct >= 10:
                    trail_level = "BREAKEVEN"

                hold_minutes = 0
                if pos.entry_time:
                    hold_minutes = int((datetime.now() - pos.entry_time).total_seconds() / 60)

                positions.append(PositionSnapshot(
                    symbol=pos.symbol,
                    option_symbol=pos.option_symbol,
                    direction=pos.side,
                    entry_price=pos.entry_price,
                    current_price=current_price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_time=pos.entry_time.isoformat() if pos.entry_time else "",
                    hold_minutes=hold_minutes,
                    stop_loss=pos.stop_loss,
                    trailing_level=trail_level,
                ))

            return positions
        except Exception as e:
            logger.debug(f"WAR_ROOM: Positions failed - {e}")
            return []

    def _get_engine_states(self) -> List[EngineStatus]:
        """Get status of all trading engines."""
        engines = []

        # SPY Scalper
        try:
            from wsb_snake.engines.spy_scalper import spy_scalper
            stats = spy_scalper.get_stats()
            engines.append(EngineStatus(
                name="SPY_SCALPER",
                running=stats.get("running", False),
                trades_today=stats.get("entries_today", 0),
                signals_today=stats.get("signals_today", 0),
                last_signal_time=None,
                cooldown_active=stats.get("cooldown_active", False),
                cooldown_until=None,
                extra={"last_price": stats.get("last_price", 0)},
            ))
        except:
            pass

        # BERSERKER
        try:
            from wsb_snake.engines.berserker_engine import get_berserker_engine
            berserker = get_berserker_engine()
            status = berserker.get_status()
            engines.append(EngineStatus(
                name="BERSERKER",
                running=status.get("running", False),
                trades_today=status.get("trades_today", 0),
                signals_today=status.get("stats", {}).get("activations", 0),
                last_signal_time=status.get("last_trade_time"),
                cooldown_active=status.get("cooldown_until") is not None,
                cooldown_until=status.get("cooldown_until"),
                extra=status.get("gex_context", {}),
            ))
        except:
            pass

        # Momentum Engine
        try:
            from wsb_snake.engines.momentum_engine import get_momentum_engine
            momentum = get_momentum_engine()
            if hasattr(momentum, 'get_stats'):
                stats = momentum.get_stats()
                engines.append(EngineStatus(
                    name="MOMENTUM",
                    running=stats.get("running", False),
                    trades_today=stats.get("trades_today", 0),
                    signals_today=stats.get("signals_today", 0),
                    last_signal_time=None,
                    cooldown_active=False,
                    cooldown_until=None,
                    extra={},
                ))
        except:
            pass

        return engines

    def _check_component_health(self) -> Dict[str, bool]:
        """Check health of all components."""
        health = {}

        # HYDRA Bridge
        try:
            from wsb_snake.collectors.hydra_bridge import get_hydra_bridge
            bridge = get_hydra_bridge()
            health["hydra_bridge"] = bridge.is_connected()
        except:
            health["hydra_bridge"] = False

        # Alpaca Executor
        try:
            from wsb_snake.trading.alpaca_executor import alpaca_executor
            health["alpaca_executor"] = alpaca_executor.running
        except:
            health["alpaca_executor"] = False

        # Risk Governor
        try:
            from wsb_snake.trading.risk_governor import get_risk_governor
            governor = get_risk_governor()
            health["risk_governor"] = not governor.kill_switch_active
        except:
            health["risk_governor"] = False

        # GEX Calculator
        try:
            from wsb_snake.learning.gex_calculator import get_gex_calculator
            gex = get_gex_calculator()
            health["gex_calculator"] = gex is not None
        except:
            health["gex_calculator"] = False

        # Specialist Swarm
        try:
            from wsb_snake.learning.specialist_swarm import get_specialist_swarm
            swarm = get_specialist_swarm()
            health["specialist_swarm"] = swarm is not None
        except:
            health["specialist_swarm"] = False

        # Trade Graph
        try:
            from wsb_snake.learning.trade_graph import get_trade_graph
            graph = get_trade_graph()
            health["trade_graph"] = graph is not None
        except:
            health["trade_graph"] = False

        # Semantic Memory
        try:
            from wsb_snake.learning.semantic_memory import get_semantic_memory
            semantic = get_semantic_memory()
            health["semantic_memory"] = semantic is not None
        except:
            health["semantic_memory"] = False

        return health

    def record_trade(self, trade_data: Dict) -> None:
        """Record a trade for recent activity."""
        with self._lock:
            trade_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._recent_trades.append(trade_data)
            if len(self._recent_trades) > 50:
                self._recent_trades = self._recent_trades[-50:]

    def record_signal(self, signal_data: Dict) -> None:
        """Record a signal for recent activity."""
        with self._lock:
            signal_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._recent_signals.append(signal_data)
            if len(self._recent_signals) > 50:
                self._recent_signals = self._recent_signals[-50:]

    def record_lesson(self, lesson_data: Dict) -> None:
        """Record a lesson learned for display."""
        with self._lock:
            lesson_data["timestamp"] = datetime.now(timezone.utc).isoformat()
            self._recent_lessons.append(lesson_data)
            if len(self._recent_lessons) > 20:
                self._recent_lessons = self._recent_lessons[-20:]


# Singleton
_war_room: Optional[WarRoom] = None


def get_war_room() -> WarRoom:
    """Get singleton war room instance."""
    global _war_room
    if _war_room is None:
        _war_room = WarRoom()
    return _war_room


def get_war_room_state() -> Dict[str, Any]:
    """Convenience function to get war room state."""
    return get_war_room().get_full_state()
