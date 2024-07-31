
document.addEventListener('DOMContentLoaded', function() {
    const anchor = window.location.hash.substring(1);
    const radio = document.getElementById(anchor);


    if (radio && radio.type === 'radio') {
        radio.checked = true;
        console.log("radio: ", radio)
    } else {
        const defaultRadio = document.getElementById('response-1');
        if (defaultRadio) defaultRadio.checked = true;
    }

    document.addEventListener('change', function(event) {
        if (event.target.type === 'radio') {
            window.location.hash = event.target.id;
        }
    });
});
