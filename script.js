function toggle_visible() {
    id_table = document.getElementById('img_id')
    id_table.classList.toggle('is-hidden')
    document.getElementById('img_ood').classList.toggle('is-hidden')

    const is_id = id_table.classList.contains('is-hidden')
    const button = document.querySelector('#id_od_toggle > span')
    button.textContent = is_id ? 'ID' : 'OOD'
    // button.classList.toggle('is-link')
    // button.classList.toggle('is-success')
}