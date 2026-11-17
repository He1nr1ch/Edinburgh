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
