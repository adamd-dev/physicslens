import { describe, expect, it } from "vitest";
import type { StreamEvent } from "./api";
import { EMPTY_RUN, applyEvent } from "./runState";

const replay = (events: StreamEvent[]) => events.reduce(applyEvent, EMPTY_RUN);

// The shape of a saved run's events, as the server streams them (only the fields the reducer reads matter).
const SAVED = [
  { event: "run", data: { id: "r1" } },
  { event: "status", data: { stage: "tracking" } },
  { event: "video", data: { frames: 60, fps: 30, scale: 1, warnings: ["slowed edit"] } },
  { event: "track", data: { notes: ["ball found automatically (motion)", "3 frames lost"] } },
  { event: "agent_message", data: { role: "assistant", text: "fit looks good" } },
  { event: "note", data: { message: "budget used up: plain fit" } },
  { event: "result", data: { status: "ok" } },
  { event: "cost", data: { run_usd: 0.012, total_usd: 1.5 } },
  { event: "done", data: {} },
] as unknown as StreamEvent[];

describe("applyEvent", () => {
  it("rebuilds a saved run from its events", () => {
    const s = replay(SAVED);
    expect(s.runId).toBe("r1");
    expect(s.stage).toBe("tracking");
    expect(s.messages).toHaveLength(1);
    expect(s.result).toEqual({ status: "ok" });
    expect(s.cost).toBe(0.012);
    expect(s.error).toBeNull();
  });

  it("keeps warnings in order and hides the auto-detect note", () => {
    expect(replay(SAVED).warnings).toEqual(["slowed edit", "3 frames lost", "budget used up: plain fit"]);
  });

  it("gives the same state for a live run and its replay", () => {
    let live = EMPTY_RUN;
    for (const e of SAVED) live = applyEvent(live, e);
    expect(replay(SAVED)).toEqual(live);
  });

  it("shows the failing stage with the error", () => {
    const s = replay([{ event: "error", data: { stage: "track", message: "ball lost" } }]);
    expect(s.error).toBe("track: ball lost");
  });

  it("ignores events it does not show, without changing state", () => {
    const s = applyEvent(EMPTY_RUN, { event: "fits", data: { fits: [] } });
    expect(s).toBe(EMPTY_RUN);
  });
});
