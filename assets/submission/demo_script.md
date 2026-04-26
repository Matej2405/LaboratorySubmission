# Demo Video script (60 seconds)

Goal: UI/UX showcase. Focus on user experience and product flow. Show that an NGO planner could actually use this tomorrow.

## Pre-flight (do this BEFORE you hit record)

1. Streamlit is already running at `http://localhost:8888`. Confirm.
2. Pre-warm the agent: open the **Ask the agent** tab, paste `Find an oncology centre in Maharashtra` into the textbox, click **Run agent**, wait for the answer to appear, then leave the result on screen.
3. Switch back to the **Crisis Map** tab. This is where you start the recording.
4. Close every other browser tab and notification. Quit Slack/Teams/Discord. Set OS to Do Not Disturb.
5. OBS canvas: 1920x1080, capture the browser window only. 30 fps is enough.

## Voiceover beats (60 sec total)

| Time | Tab / View | What you do | What you say |
|---|---|---|---|
| 0 - 7 s | Crisis Map (already open) | Hold on the choropleth | "70% of India lives in rural areas where finding a hospital with the right equipment is a discovery crisis. We turn 10,000 messy facility reports into this." |
| 7 - 18 s | Crisis Map | Slowly hover 2-3 districts so the tooltip shows desert score + Wilson interval | "Every district has a population-aware desert score for any high-acuity capability you pick - ICU, dialysis, oncology - bounded by a Wilson 95% interval, so you see uncertainty, not a guessy number." |
| 18 - 30 s | Crisis Map - scroll to Featured Findings | Stop on the three smoking-gun cards | "The validator agent surfaces three smoking guns: ICU claimed but no ventilator documented, with the verbatim sentence cited - Chiguru Child Care, Aphila, and Arihant. An NGO can audit a claim in two clicks." |
| 30 - 41 s | Trust audit tab | Click the tab, scroll past the bar chart to the flag list | "Across the full 10k, fourteen contradiction classes are flagged with the source sentence and the row ID. No black box - every flag links back to the verbatim text." |
| 41 - 60 s | Ask the agent tab | Click the tab, the pre-warmed Maharashtra answer is visible. Click **Run agent** again to refresh, scroll the citations | "And the agent itself: plan, retrieve, cite, compose, all traced in MLflow. 'Find an oncology centre in Maharashtra' returns 10 facilities, ranked by trust, with the exact sentence from the original note that justifies each claim. That cuts discovery-to-care time from hours to seconds." |

## Visual polish

- Use Windows display scaling 125% so tooltip text is readable in the recording.
- Move the mouse with intent - hover, dwell, click. Avoid jittery cursor motion.
- Do NOT show the OS taskbar. Use full-screen browser if possible (F11 in Chrome).
- If you fluff a take, cut and re-record only that segment - OBS lets you stitch.

## Save as

`assets/submission/demo.mp4` (mp4, H.264, max 60 sec, target 20-40 MB).
