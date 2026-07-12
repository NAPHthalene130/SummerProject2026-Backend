from fastapi import APIRouter, HTTPException, status

from app.models.user import UserCreateRequest, UserResponse, UserUpdateRequest
from app.repository.user_repository import UserNameAlreadyExistsError, UserRepository
from app.utils.passwords import hash_password


users_router = APIRouter()


@users_router.get("/", response_model=list[UserResponse])
async def list_users():
    return UserRepository.list_users()


@users_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(request: UserCreateRequest):
    try:
        return UserRepository.create_user(
            user_name=request.user_name,
            user_password=hash_password(request.password),
            user_type=request.user_type,
        )
    except UserNameAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="用户名已存在") from exc


@users_router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: int, request: UserUpdateRequest):
    try:
        user = UserRepository.update_user(
            user_id=user_id,
            user_name=request.user_name,
            user_type=request.user_type,
            user_password=hash_password(request.password) if request.password else None,
        )
    except UserNameAlreadyExistsError as exc:
        raise HTTPException(status_code=409, detail="用户名已存在") from exc

    if user is None:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


@users_router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    if not UserRepository.delete_user(user_id):
        raise HTTPException(status_code=404, detail="用户不存在")
