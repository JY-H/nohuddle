import cv2
import numpy as np

from nohuddle import gameclock, scheduler
from nohuddle.detector import BreakDetector, State
from nohuddle.gameclock import GameClock, Reading, Text, parse
from nohuddle.health import PlaybackHealth
from nohuddle.streams import deobfuscate

rng = np.random.default_rng(0)


def scene(t: float) -> np.ndarray:
    """A moving, textured 'camera' picture that changes every frame."""
    img = rng.integers(0, 255, (36, 64, 3), dtype=np.uint8)
    return cv2.resize(img, (640, 360), interpolation=cv2.INTER_CUBIC)


def field(t: float) -> np.ndarray:
    """Game footage: changes every frame, but always a green field (unlike ads)."""
    img = scene(t)
    img[:, :, 1] = np.clip(img[:, :, 1].astype(int) // 2 + 110, 0, 255)
    img[:, :, 0] //= 3
    img[:, :, 2] //= 3
    return img


def broadcast(t: float) -> np.ndarray:
    img = field(t)
    cv2.rectangle(img, (150, 300), (490, 340), (20, 20, 20), -1)  # scorebug
    cv2.putText(img, "DUKE 3  2ND  MIAMI 9", (160, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "NET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  # network logo
    return img


def logo_only(t: float) -> np.ndarray:
    img = field(t)
    cv2.putText(img, "NET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img


def run(det, frames, t0, t1):
    st = None
    for t in np.arange(t0, t1, 1.0):
        st = det.observe(frames(t), float(t))
    return st


def test_detects_break_and_return():
    det = BreakDetector(read_clock=False)
    assert run(det, broadcast, 0, 120) == State.LIVE
    assert run(det, scene, 120, 123) == State.LIVE  # too short to be a break
    assert run(det, scene, 123, 200) == State.BREAK
    assert run(det, broadcast, 200, 203) == State.LIVE


def test_region_boxes_are_json_serialisable():
    import json
    det = BreakDetector(read_clock=False)
    run(det, broadcast, 0, 120)
    json.dumps(det.model.boxes())


def test_network_logo_alone_keeps_it_live():
    """Scorebug hidden (side panel, replay graphic) but the network logo is still there."""
    det = BreakDetector(read_clock=False)
    run(det, broadcast, 0, 120)
    assert run(det, logo_only, 120, 160) == State.LIVE


def test_long_break_does_not_erase_overlays():
    det = BreakDetector(read_clock=False)
    run(det, broadcast, 0, 180)
    assert run(det, scene, 180, 420) == State.BREAK  # 4 minutes of ads
    assert run(det, broadcast, 420, 424) == State.LIVE


def test_halftime_show_with_only_the_logo_is_a_break():
    det = BreakDetector(read_clock=False)
    run(det, broadcast, 0, 180)
    assert run(det, logo_only, 180, 290) == State.LIVE  # could just be a long replay package
    assert run(det, logo_only, 290, 600) == State.BREAK
    assert det.detail == "halftime"
    # Clock unreadable, so halftime lasts at least the minimum (12 min from ~300s) even if the
    # scorebug shows up again (e.g. in highlights)...
    assert run(det, broadcast, 600, 700) == State.BREAK
    # ...then the scorebug has to stay up for 30s.
    assert run(det, broadcast, 1020, 1060) == State.LIVE


def test_starting_during_halftime_show():
    """Started mid-halftime: nothing to learn at first, then the game comes back."""
    det = BreakDetector(read_clock=False)
    assert run(det, scene, 0, 100) == State.LEARNING
    assert run(det, scene, 100, 400) == State.BREAK
    assert run(det, broadcast, 400, 560) == State.LIVE


def test_manual_override():
    det = BreakDetector(read_clock=False)
    run(det, broadcast, 0, 120)
    det.force(State.BREAK, 120)
    assert run(det, broadcast, 120, 130) == State.BREAK
    det.force(State.LIVE, 130)  # "not a break" on a manual break just cancels it
    assert run(det, broadcast, 130, 135) == State.LIVE
    run(det, scene, 135, 150)
    det.force(State.LIVE, 150)  # "not a break" on a detected break: relearn
    assert run(det, scene, 150, 170) in scheduler.WATCHABLE


def test_override_sticks_while_learning():
    det = BreakDetector(read_clock=False)
    run(det, broadcast, 0, 20)
    det.force(State.BREAK, 20)
    assert run(det, broadcast, 20, 30) == State.BREAK


def test_scheduler_prefers_priority_and_never_shows_breaks():
    p = ["a", "b", "c"]
    assert scheduler.choose(p, {"a", "b"}, None) == "a"
    assert scheduler.choose(p, {"c"}, None) == "c"
    assert scheduler.choose(p, set(), None) is None  # everything on a break or unconfirmed
    assert scheduler.choose(p, {"a"}, "c") == "c"


def test_learning_games_are_hidden_until_confirmed(monkeypatch):
    shown = {"text": []}
    monkeypatch.setattr(gameclock, "read_text", lambda frame: shown["text"])
    readable = BreakDetector(read_clock=True)
    run(readable, broadcast, 0, 4)
    assert readable.state == State.LEARNING and not readable.showable(4)
    shown["text"] = texts("DUKE", "2ND 5:00", "MIAMI")
    run(readable, broadcast, 4, 10)
    assert readable.showable(10)  # first good readings
    unreadable = BreakDetector(read_clock=False)
    run(unreadable, broadcast, 0, 30)
    assert not unreadable.showable(30)
    run(unreadable, broadcast, 30, 50)
    assert unreadable.showable(50)  # can't confirm it, so after the grace period show it anyway


def test_health_lag_and_stall():
    h = PlaybackHealth()
    for i in range(40):  # smooth playback
        h.update(i * 2.0, i * 2.0, 0)
    assert not h.laggy() and not h.stalled(80)
    h = PlaybackHealth()
    for i in range(40):  # plays half as fast as real time
        h.update(i * 2.0, i * 1.0, 0)
    assert h.laggy()
    h = PlaybackHealth()
    for i in range(40):  # keeps rebuffering
        h.update(i * 2.0, i * 2.0, i // 5)
    assert h.laggy()
    h = PlaybackHealth()
    h.update(0, 5.0, 0)
    h.update(20, 5.0, 0)
    assert h.stalled(20)


def test_deobfuscate():
    assert deobfuscate("Jzs7Pzx1YGAqNy4iPyMqYSwgImAqIi0qK2AsKS1gKC4iKg==") == \
        "https://example.com/embed/cfb/game"


def texts(*strings):
    return [Text(x, 0.5, 0.9) for x in strings]


def test_parse_scorebug_text():
    r = parse(texts("DUKE", "3", "2N0 5:53", "2ND & GOAL", "7 MIAMI"))
    assert (r.quarter, r.clock) == (2, 353)
    assert "DUKE" in r.words and "MIAMI" in r.words
    assert parse(texts("2KD", "2:51")).quarter == 2  # OCR-mangled quarter
    assert parse(texts("1ST & 10", "12:00")).quarter is None  # down and distance, not a quarter
    assert parse(texts("4th. &.11", "6:38")).quarter is None
    assert parse(texts("2nd.84", "2:13")).quarter is None  # "2nd & 4"
    assert parse(texts("2ND 8:47")).quarter == 2
    assert parse(texts("13", "6:3831")).clock == 398  # clock run into the play clock
    assert parse(texts("3RD", "6.34")).clock == 394
    assert (parse(texts("DUKE", "3Ro :47", "19")).quarter, parse(texts("3Ro :47")).clock) == (3, 47)  # under a minute
    assert parse(texts("13", ":25")).clock is None  # a lone play clock
    # ESPN, under a minute, with other text between the quarter and the clock (from a report)
    assert parse(texts("12 TEXAS TECH", "1sт", "ESFT", ":53", "30", "25 OKLAHOMA ST")).clock == 53
    drive = parse(texts("ESPN", "SCORING DRIVE", "16", "PLAYS YARDS", "2:20", "TIME", "20", "1ST", ":53"))
    assert drive.clock == 53  # not the drive's 2:20
    assert parse(texts("2ND & GOAL", "3RD", ":47", "34")).clock == 47
    assert parse(texts("FOX NFL", "35", ":15", "2ND")).clock == 15  # FOX: clock before the quarter
    # ABC: "1st & 10" read as "1Sт 8.10" (Cyrillic t, & as 8) next to the real clock
    abc = parse(texts("SEC ON ABC", "22 MISSOURI", "1Sт 8.10", "3:32", "1 GEORGIA"))
    assert abc.quarter is None and abc.clock == 212
    assert parse(texts("2ND 8.47")).clock == 527  # no other clock: still a (misread) clock
    assert parse(texts("3RD", "6:33", ":25")).clock == 393  # full clock wins over the play clock
    assert parse(texts("3:30 PM ET")).clock is None
    assert parse(texts("HALFTIME")).halftime
    far = parse([Text("3RD", 0.1, 0.9), Text("8:12", 0.9, 0.1)])
    assert far.quarter == 3 and far.clock is None  # clock on the other side of the screen


def reading(q, clock, words=("DUKE", "MIAMI")):
    return Reading(q, clock, words=set(words))


def test_clock_flags_highlights_from_earlier():
    c = GameClock()
    for i in range(10):
        c.update(reading(2, 300 - i * 5), i * 2.0)
    assert c.verdict(20) is None
    c.update(reading(2, 260), 22)  # replay of the play that just happened: fine
    assert c.verdict(22) is None
    c.update(reading(1, 600), 24)  # first-quarter highlight: one reading could be a misread...
    assert c.verdict(24) is None
    c.update(reading(1, 598), 26)  # ...two consistent ones are a highlight
    assert c.verdict(26) == "highlight"


def test_clock_halftime_until_third_quarter():
    c = GameClock()
    for i in range(5):
        c.update(reading(2, 10 - 2 * i if i < 5 else 0), i * 2.0)
    c.update(reading(2, 0), 12)
    c.update(reading(2, 0), 14)
    assert c.in_halftime(14, 0)
    c.update(reading(1, 400), 300)  # halftime highlights
    assert c.in_halftime(300, 30)
    c.update(reading(3, 900), 900)
    c.update(reading(3, 895), 904)
    assert not c.in_halftime(904, 4)


def test_clock_other_games_highlights():
    c = GameClock()
    for i in range(20):
        c.update(reading(3, 800 - i * 5), i * 2.0)
    c.update(reading(4, 200, words=("CHIEFS", "BILLS")), 50)  # far ahead, wrong teams
    c.update(reading(4, 198, words=("CHIEFS", "BILLS")), 52)
    assert c.verdict(52) == "other game"
    c.update(reading(3, 700, words=("HOLDING",)), 52)  # penalty banner over our own scorebug
    assert c.verdict(52) is None


def test_highlights_during_play_stay_on_screen(monkeypatch):
    shown = {"text": texts("DUKE", "2ND 5:00", "MIAMI")}
    monkeypatch.setattr(gameclock, "read_text", lambda frame: shown["text"])
    det = BreakDetector(read_clock=True)
    assert run(det, broadcast, 0, 120) == State.LIVE
    shown["text"] = texts("DUKE", "1ST 10:00", "MIAMI")  # a 1st-quarter highlight, mid 2nd quarter
    assert run(det, broadcast, 120, 140) == State.LIVE  # part of the broadcast: keep watching
    shown["text"] = texts("DUKE", "2ND 4:50", "MIAMI")
    assert run(det, broadcast, 140, 145) == State.LIVE


def test_highlights_at_halftime_stay_hidden(monkeypatch):
    shown = {"text": texts("DUKE", "2ND 0:05", "MIAMI")}
    monkeypatch.setattr(gameclock, "read_text", lambda frame: shown["text"])
    det = BreakDetector(read_clock=True)
    run(det, broadcast, 0, 120)  # end of the first half (halftime not confirmed on screen)
    shown["text"] = texts("DUKE", "1ST 10:00", "MIAMI")  # halftime show replaying the 1st quarter
    assert run(det, broadcast, 120, 140) == State.BREAK
    assert det.detail == "highlight"


def test_wrong_model_is_relearned_when_clock_says_live(monkeypatch):
    shown = {"text": []}
    monkeypatch.setattr(gameclock, "read_text", lambda frame: shown["text"])
    det = BreakDetector(read_clock=True)
    run(det, logo_only, 0, 120)  # learned only a bumper's "logo"
    assert det.model and det.model.regions
    shown["text"] = texts("DUKE", "3RD 12:00", "MIAMI")  # the real game: different look
    states = [det.observe(scene(t), float(t)) for t in range(120, 160)]
    assert det.relearned == 1
    assert states[-1] in scheduler.WATCHABLE


def feed(c, q, start_clock, t0, seconds, step=2.0, words=("DUKE", "MIAMI")):
    """Readings of a live clock counting down in real time."""
    t = t0
    while t < t0 + seconds:
        c.update(reading(q, max(0, int(start_clock - (t - t0))), words), t)
        t += step
    return t


def test_other_games_clock_during_halftime_cannot_move_us_forward():
    c = GameClock()
    feed(c, 2, 120, 0, 126)  # end of the 2nd quarter
    assert c.max_elapsed == 1800
    feed(c, 3, 115, 300, 30, words=())  # a look-in at another game: "3RD 1:55"
    assert c.max_elapsed == 1800
    assert c.verdict(330) in (None, "other game")  # never "live"
    feed(c, 3, 900, 1500, 20)  # our 3rd quarter kicks off
    assert c.live(1518) and c.display.startswith("Q3")


def test_wrong_position_corrects_itself():
    c = GameClock()
    feed(c, 3, 300, 0, 10)
    c.max_elapsed, c.max_at = 2585, 10  # e.g. restored from a bad save
    t = feed(c, 3, 540, 20, 30)  # real game at Q3 9:00: looks like a highlight at first...
    assert c.verdict(t - 2) == "highlight"
    t = feed(c, 3, 510, t, 40)  # ...but a clock that keeps running steadily is live play
    assert c.verdict(t - 2) is None and c.live(t - 2)
    assert c.max_elapsed < 2585


def test_quarter_carried_over_when_unreadable():
    c = GameClock()
    feed(c, 3, 400, 0, 10)
    c.update(Reading(None, 392, words={"DUKE"}), 12)
    assert c.display == "Q3 6:32"


def test_misread_quarter_on_the_running_clock_is_corrected():
    c = GameClock()
    t = feed(c, 3, 140, 0, 20)  # Q3 2:20 -> 2:02
    c.update(reading(2, 120), t)  # "2nd & 4" misread as the quarter, clock still 2:00
    assert c.verdict(t) is None and c.display == "Q3 2:00"
    c.update(reading(1, 400), t + 2)  # a real 1st-quarter highlight
    c.update(reading(1, 398), t + 4)
    assert c.verdict(t + 4) == "highlight"


def test_known_network_gives_new_games_a_head_start(tmp_path):
    from nohuddle.detector import DetectorSettings
    from nohuddle.networks import NetworkLibrary

    settings = DetectorSettings()
    first = BreakDetector(settings, read_clock=False)
    run(first, broadcast, 0, 120)  # a game on this network, watched to the end of learning
    library = NetworkLibrary(tmp_path, settings)
    assert library.learn(first.model.persistence, "NET") == "NET"

    reloaded = NetworkLibrary(tmp_path, settings)  # survives restarts
    second = BreakDetector(settings, read_clock=False, networks=reloaded.candidates)
    assert run(second, broadcast, 0, 8) == State.LIVE  # no minute of learning needed
    assert second.network == "NET"
    assert run(second, scene, 8, 30) == State.BREAK  # and it already knows what an ad looks like


def test_library_keeps_different_networks_apart(tmp_path):
    from nohuddle.detector import DetectorSettings
    from nohuddle.networks import NetworkLibrary

    settings = DetectorSettings()
    library = NetworkLibrary(tmp_path, settings)
    a = BreakDetector(settings, read_clock=False)
    run(a, broadcast, 0, 120)
    b = BreakDetector(settings, read_clock=False)

    def other_network(t):
        img = scene(t)
        cv2.rectangle(img, (40, 20), (200, 70), (200, 30, 30), -1)
        cv2.putText(img, "OTHER 7 - 3", (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    run(b, other_network, 0, 120)
    assert library.learn(a.model.persistence, "NET") == "NET"
    assert library.learn(b.model.persistence) != "NET"
    assert library.learn(a.model.persistence, "NET") == "NET"
    assert library.networks["NET"].games == 2


def test_learning_game_with_readable_clock_hides_ads(monkeypatch):
    """Started (or restarted) during an ad break, before any overlays are learned."""
    shown = {"text": texts("TEXAS A&M", "2nd QTR 12:36", "TENNESSEE")}
    monkeypatch.setattr(gameclock, "read_text", lambda frame: shown["text"])
    det = BreakDetector(read_clock=True)
    assert run(det, broadcast, 0, 20) in scheduler.WATCHABLE  # clock confirms the game
    shown["text"] = []
    assert run(det, scene, 20, 50) == State.BREAK  # ads: hidden ~12s in, instead of minutes
    assert all(t <= 32 for t, _ in det.history)  # and mostly not learned from
    shown["text"] = texts("TEXAS A&M", "2nd QTR 12:30", "TENNESSEE")
    assert run(det, broadcast, 50, 56) in scheduler.WATCHABLE


def test_single_reading_resumes_where_the_game_was():
    c = GameClock()
    feed(c, 2, 760, 0, 10)  # Q2 12:40 -> 12:32
    assert c.max_elapsed is not None
    c.update(reading(2, 756), 200)  # back from ads, one reading after a long gap
    assert c.live(200)
    c.update(reading(4, 300), 300)  # a lone reading that jumps: still needs a second one
    assert not c.live(300)


def test_quarter_inferred_from_position_when_unreadable():
    c = GameClock()
    feed(c, 3, 470, 0, 10)  # Q3 7:50 -> 7:42, then the quarter stops being readable
    t = 300.0
    for clock in range(228, 200, -2):  # "3:48", "3:46", ... with no quarter
        c.update(Reading(None, clock, words={"GEORGIA"}), t)
        t += 2
    assert c.live(t - 2) and c.display == "Q3 3:22"


def test_position_cannot_run_ahead_of_kickoff():
    from nohuddle.gameclock import ClockSettings
    kickoff = 1000.0
    c = GameClock(ClockSettings(kickoff=kickoff))
    t = kickoff + 25 * 60  # 25 minutes after kickoff: early 1st quarter at best...
    c.update(reading(3, 462), t)  # ..."3RD 8.5" + the real clock: impossible, ignored
    c.update(reading(3, 460), t + 2)
    assert c.max_elapsed is None and not c.halftime_done
    c.max_elapsed, c.max_at, c.halftime_done = 2238, t, True  # a bad position saved earlier
    feed(c, 1, 202, t + 4, 10)  # the real game: Q1 3:22
    assert c.max_elapsed < 900 and not c.halftime_done
    c.halftime_done = True  # a bad flag that survived (the position was plausible by then)
    feed(c, 1, 190, t + 20, 6)
    assert not c.halftime_done
    assert c.live(t + 12) and c.display.startswith("Q1")


def test_halftime_stats_panel_is_not_the_second_half():
    # From a report: "3rd DOWNS" read as the quarter, time of possession as the clock.
    panel = texts("6 TEXAS", "10", "14", "21LSU", "TEXAS", "LSU", "144", "TOTAL YARDS", "206", "3-9",
                  "3rd DOWNS", "5-7", "TURNOVERS", "14:56", "15:04")
    assert parse(panel).clock is None
    assert parse(texts("3RD DOWNS", "14:56")).quarter is None
    c = GameClock()
    feed(c, 2, 20, 0, 24)  # end of the 2nd quarter
    c.update(reading(2, 0), 30)
    c.update(reading(2, 0), 32)
    assert c.in_halftime(32, 0)
    for k in range(10):  # a frozen "3RD 15:00" during the halftime show
        c.update(reading(3, 900), 600 + 2 * k)
    assert c.in_halftime(620, 0)
    feed(c, 3, 900, 1500, 10)  # kickoff: the clock runs
    assert not c.in_halftime(1510, 0) and c.live(1508)


def test_basketball_replay_without_overlays_stays_on_screen():
    from nohuddle.sports import NBA
    det = BreakDetector(read_clock=False, extras=NBA.make_extras())
    run(det, broadcast, 0, 150)
    assert run(det, field, 150, 165) == State.LIVE  # 15s replay: no scorebug or logo, same game
    assert run(det, broadcast, 165, 170) == State.LIVE
    assert run(det, scene, 170, 180) == State.BREAK  # an actual ad still goes to break
    run(det, broadcast, 180, 200)
    assert run(det, field, 200, 260) == State.BREAK  # but not forever


def test_ad_blip_does_not_flip_a_readable_game_live(monkeypatch):
    """From a report: an ad briefly matched an overlay, then its colours were learned as the game's."""
    shown = {"text": texts("LSU", "3RD 9:51", "TEXAS")}
    monkeypatch.setattr(gameclock, "read_text", lambda frame: shown["text"])
    det = BreakDetector(read_clock=True)
    run(det, broadcast, 0, 150)
    shown["text"] = []  # commercial break: no clock
    assert run(det, scene, 150, 200) == State.BREAK
    assert run(det, broadcast, 200, 203) == State.BREAK  # overlays for 3s in an ad: not enough
    assert run(det, scene, 203, 240) == State.BREAK  # and it didn't learn the ad's colours
    shown["text"] = texts("LSU", "3RD 9:50", "TEXAS")  # the game is back, clock and all
    assert run(det, broadcast, 240, 244) == State.LIVE


def test_basketball_rule_is_not_used_for_football():
    from nohuddle.basketball import ReplayByLook
    from nohuddle.sports import NFL, COLLEGE_FOOTBALL
    for sport in (NFL, COLLEGE_FOOTBALL):
        assert not any(isinstance(x, ReplayByLook) for x in sport.make_extras())


def test_football_highlight_on_the_field_stays_on_screen():
    from nohuddle.sports import NFL
    det = BreakDetector(read_clock=False, extras=NFL.make_extras())
    run(det, broadcast, 0, 150)
    assert run(det, field, 150, 175) == State.LIVE  # replay/highlight: no overlays, but a field
    assert run(det, broadcast, 175, 180) == State.LIVE
    assert run(det, scene, 180, 190) == State.BREAK  # an ad (not field green) still goes to break
    run(det, broadcast, 190, 210)
    assert run(det, field, 210, 270) == State.BREAK  # and the allowance runs out


def test_nba_clock_uses_12_minute_quarters():
    from nohuddle.gameclock import ClockSettings
    from nohuddle.sports import NBA
    c = GameClock(ClockSettings(period_seconds=NBA.period_seconds, periods_per_half=NBA.periods_per_half))
    feed(c, 2, 30, 0, 34)  # end of the 2nd quarter
    assert c.max_elapsed == 2 * 720  # halftime at 24 minutes of game clock, not 30
    c.update(reading(2, 0), 40)
    c.update(reading(2, 0), 42)
    assert c.in_halftime(42, 0)
    feed(c, 3, 720, 900, 10)  # 3rd quarter tips off at 12:00
    assert not c.in_halftime(910, 0) and c.display.startswith("Q3 11:")


def test_saved_position_with_other_period_length_is_ignored():
    from nohuddle.gameclock import ClockSettings
    old = GameClock()  # saved before NBA quarters were 12 minutes
    old.max_elapsed = 2285
    nba = GameClock(ClockSettings(period_seconds=720))
    nba.load(old.to_json())
    assert nba.max_elapsed is None
    same = GameClock()
    same.load(old.to_json())
    assert same.max_elapsed == 2285


def test_return_after_a_score_with_graphics_flickering(monkeypatch):
    """From a report: after a touchdown, graphics kept covering the scorebug, so the overlays were
    never up 6s without a gap and the game never came back."""
    monkeypatch.setattr(gameclock, "read_text", lambda frame: texts("TEXAS TECH", "1ST 1:20", "OKLAHOMA ST"))
    det = BreakDetector(read_clock=True)
    run(det, broadcast, 0, 150)
    monkeypatch.setattr(gameclock, "read_text", lambda frame: [])  # clock unreadable from here on
    assert run(det, scene, 150, 170) == State.BREAK
    states = [det.observe(broadcast(t) if t % 5 else scene(t), float(t)) for t in range(170, 190)]
    assert states[-1] == State.LIVE  # overlays up 80% of the time: back on screen


def test_long_break_mid_second_quarter_is_not_halftime():
    """From a report: a 5+ minute break at Q2 8:25 was taken for halftime."""
    c = GameClock()
    feed(c, 2, 515, 0, 10)  # Q2 8:35 -> 8:25
    c.note_intermission(400)  # long break
    assert not c.in_halftime(400, 0)
    c2 = GameClock()
    feed(c2, 2, 100, 0, 10)  # Q2 1:40: a long break now is halftime
    c2.note_intermission(400)
    assert c2.in_halftime(400, 0)
    c2.reset_position(410)  # "Not a break"
    assert not c2.in_halftime(410, 0)
