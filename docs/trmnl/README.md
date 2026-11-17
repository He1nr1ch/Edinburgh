# Edinburgh on the TRMNL

A TRMNL screenshot of the dashboard page comes out blank: the page fills itself in with
JavaScript after it loads, and its dark theme is lost on e-ink. Use a Private Plugin
instead. TRMNL fetches the dashboard's JSON and draws it with its own e-ink styles.

1. On trmnl.com: **Plugins → Private Plugin → Add new**.
2. **Strategy:** Polling.
3. **Polling URL:** `https://lionfish-app-jbbar.ondigitalocean.app/api/accounts`.
4. **Refresh rate:** every 15 minutes is enough; the feed's prices are live while the market is open.
5. Save, then **Edit Markup → Full** and paste in [`full.liquid`](full.liquid).
6. Add the plugin to the device's playlist.

The template shows the two Edinburgh accounts, Heinrich and Valya, with SPY and IWM
along the bottom. To show Letterman or RUSS2K as well, add their names to the `if` near
the top of the template.

## The open positions

The positions page lives on Colossus, which TRMNL's servers cannot reach, so the worker
sends the positions out instead: after every pass it posts them to a second Private
Plugin, and [`positions.liquid`](positions.liquid) lays them out, one row a position
with its P&L and its distance to the stop and to the target.

1. On trmnl.com: **Plugins → Private Plugin → Add new**, **Strategy:** Webhook. Save.
2. Copy the **Webhook URL** the plugin now shows.
3. **Edit Markup → Full**, paste in [`positions.liquid`](positions.liquid).
4. On Colossus, add a line to the worker's `.env` and start the worker again:
   `TRMNL_WEBHOOK_URL=<the webhook URL>`
5. Add the plugin to the device's playlist.

The screen changes when a position or a figure on it changes; a pass that finds nothing
new sends nothing. TRMNL takes 12 posts an hour, which is the worker's own pace.
The template has room for about 24 positions.
