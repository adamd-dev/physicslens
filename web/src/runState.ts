import { useCallback, useState } from "react";
import type { AgentMessage, Detection, Result, StreamEvent, TrackEvent } from "./api";

export interface RunState {
  runId: string | null;
  stage: string | null;
  track: TrackEvent | null;
  messages: AgentMessage[];
  result: Result | null;
  cost: number | null;
  error: string | null;
  warnings: string[];
  detection: Detection | null;
}

export const EMPTY_RUN: RunState = {
  runId: null, stage: null, track: null, messages: [], result: null, cost: null, error: null, warnings: [], detection: null,
};

/** One event applied to the run state: the live stream and a saved run's replay share it, so History shows exactly what was seen. */
export function applyEvent(s: RunState, e: StreamEvent): RunState {
  switch (e.event) {
    case "run":
      return { ...s, runId: e.data.id };
    case "note":
      return { ...s, warnings: [...s.warnings, e.data.message] };
    case "status":
      return { ...s, stage: e.data.stage };
    case "video":
      return { ...s, warnings: e.data.warnings };
    case "detection":
      return { ...s, detection: e.data };
    case "track":
      // The auto-detect note is shown as information next to the detection, not as a warning.
      return { ...s, track: e.data, warnings: [...s.warnings, ...(e.data.notes ?? []).filter((n) => !n.startsWith("ball found automatically"))] };
    case "agent_message":
      return { ...s, messages: [...s.messages, e.data] };
    case "result":
      return { ...s, result: e.data };
    case "cost":
      return { ...s, cost: e.data.run_usd };
    case "error":
      return { ...s, error: `${e.data.stage}: ${e.data.message}` };
    default:
      return s;
  }
}

export function useRunState() {
  const [state, setState] = useState<RunState>(EMPTY_RUN);
  const onEvent = useCallback((e: StreamEvent) => setState((s) => applyEvent(s, e)), []);
  const reset = useCallback(() => setState(EMPTY_RUN), []);
  const setError = useCallback((error: string | null) => setState((s) => ({ ...s, error })), []);
  return { state, onEvent, reset, setError };
}
