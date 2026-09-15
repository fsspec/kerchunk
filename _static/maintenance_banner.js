(function () {
    var DISMISS_KEY = "kerchunkMaintenanceBannerDismissed";

    try {
        if (localStorage.getItem(DISMISS_KEY) === "1") {
            return;
        }
    } catch (e) {
        // localStorage unavailable (e.g. private browsing); show the banner anyway.
    }

    document.addEventListener("DOMContentLoaded", function () {
        var banner = document.createElement("div");
        banner.id = "kerchunk-maintenance-banner";

        var content = document.createElement("div");
        content.id = "kerchunk-maintenance-banner-content";
        content.innerHTML =
            '<strong id="kerchunk-maintenance-banner-title">Kerchunk is in maintenance mode: no new features are planned, ' +
            "though bug fixes will continue to be accepted.</strong><br><br>" +
            "For new projects, we recommend " +
            '<a href="https://virtualizarr.readthedocs.io/">VirtualiZarr</a> for creating ' +
            "virtual Zarr datasets, together with " +
            '<a href="https://icechunk.io/">Icechunk</a> as the storage engine.<br><br>' +
            'See the FAQ for a <a href="https://virtualizarr.readthedocs.io/en/stable/explanation/faq.html#how-do-the-virtualizarr-and-kerchunk-libraries-compare">comparison of libraries</a>, ' +
            '<a href="https://virtualizarr.readthedocs.io/en/stable/explanation/faq.html#which-format-should-i-save-my-virtual-references-as">comparison of supported storage formats</a>, ' +
            "and for how to " +
            '<a href="https://virtualizarr.readthedocs.io/en/stable/explanation/faq.html#i-have-already-kerchunked-my-data-do-i-have-to-redo-that">migrate existing Kerchunk references</a>.';
        banner.appendChild(content);

        var dismiss = document.createElement("button");
        dismiss.setAttribute("aria-label", "Dismiss");
        dismiss.textContent = "×";
        dismiss.addEventListener("click", function () {
            banner.remove();
            try {
                localStorage.setItem(DISMISS_KEY, "1");
            } catch (e) {
                // ignore
            }
        });
        banner.appendChild(dismiss);

        document.body.insertBefore(banner, document.body.firstChild);
    });
})();
