pub trait UnwrapHelper {
    type Item;

    fn unwrap_or_exit_with_message(self) -> Self::Item;
}

impl<T> UnwrapHelper for Result<T, String> {
    type Item = T;

    fn unwrap_or_exit_with_message(self) -> T {
        self.unwrap_or_else(|message| {
            eprintln!("\n{message}");
            std::process::exit(1);
        })
    }
}
