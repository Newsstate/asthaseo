// Toggle category collapse/expand
function toggleCategory(category) {
    const body = document.getElementById(`body-${category}`);
    const toggle = body.previousElementSibling.querySelector(".toggle-button");
    if (body.style.display === "block") {
        body.style.display = "none";
        toggle.textContent = "+";
    } else {
        body.style.display = "block";
        toggle.textContent = "-";
    }
}

// Search/filter functionality
document.getElementById("searchInput").addEventListener("keyup", function() {
    const filter = this.value.toLowerCase();
    const cards = document.getElementsByClassName("category-card");

    for (let card of cards) {
        const items = card.getElementsByTagName("li");
        let matchFound = false;

        for (let item of items) {
            if (item.textContent.toLowerCase().indexOf(filter) > -1) {
                item.style.display = "";
                matchFound = true;
            } else {
                item.style.display = "none";
            }
        }

        // Show/hide entire category if it has matches
        card.style.display = matchFound ? "" : "none";
    }
});
